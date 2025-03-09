"""
BioKANモデルの中核実装
コルモゴロフ・アーノルド・ネットワークを拡張して生体模倣的アーキテクチャを提供
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

from biokan.core.layers import TernaryActivationFunction, MultiHeadAttention, KANLinear
from biokan.neuro.neuromodulators import NeuromodulatorSystem
from biokan.neuro.glial_cells import Astrocyte, Microglia


class BiologicalMultiHeadAttention(MultiHeadAttention):
    """神経学的特性を持つマルチヘッドアテンション層"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, neuromodulation=True):
        """
        初期化
        
        Args:
            embed_dim: 埋め込み次元
            num_heads: ヘッド数
            dropout: ドロップアウト確率
            neuromodulation: 神経調節を有効にするかどうか
        """
        super().__init__(embed_dim, num_heads, dropout)
        
        # 神経調節システム（オプション）
        self.neuromodulator = NeuromodulatorSystem() if neuromodulation else None
        
        # アストロサイト活動（オプション）
        if neuromodulation:
            self.astrocyte = Astrocyte(region_shape=(num_heads, embed_dim // num_heads))
        else:
            self.astrocyte = None
        
        # ヘッドごとの活性化閾値（これはニューロンの発火閾値に相当）
        self.head_thresholds = nn.Parameter(torch.ones(num_heads) * 0.1)
        
        # アテンション重みキャッシュ（説明可能性のため）
        self.attention_weights = None
    
    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        """
        フォワードパス
        
        Args:
            query: クエリテンソル
            key: キーテンソル
            value: バリューテンソル
            attn_mask: アテンションマスク
            need_weights: 重みを返すかどうか
            
        Returns:
            出力テンソルとオプションのアテンション重み
        """
        # 基本的なアテンション計算
        attn_output, attn_weights = super().forward(query, key, value, attn_mask, True)
        
        # アテンション重みをキャッシュ
        self.attention_weights = attn_weights.detach()
        
        # 神経調節システムがある場合
        if self.neuromodulator is not None:
            # アテンション活動に基づく神経調節の更新
            attn_activity = attn_weights.mean(dim=1).mean(dim=0)  # ヘッドごとの平均アテンション
            
            # 神経伝達物質レベルの更新
            neuromodulator_effects = {
                'dopamine': 0.2 * attn_activity.max().item(),       # 最大アテンションに応じた報酬
                'acetylcholine': 0.1 * attn_activity.mean().item(), # 平均アテンションに応じた注意
                'serotonin': -0.05 * (attn_weights.var().item()),   # アテンションのばらつきが大きいと抑制的
                'noradrenaline': 0.15 if attn_activity.max().item() > 0.7 else -0.05  # 強いアテンションで覚醒
            }
            
            self.neuromodulator.update(stimuli=neuromodulator_effects)
            
            # 現在の神経伝達物質状態を取得
            neuro_state = self.neuromodulator.get_state()
            
            # アテンション出力の調整
            dopamine_effect = 1.0 + 0.2 * neuro_state['dopamine']  # ドーパミンによる信号増幅
            serotonin_effect = 1.0 - 0.1 * neuro_state['serotonin'] if neuro_state['serotonin'] > 0 else 1.0 + 0.05 * abs(neuro_state['serotonin'])
            
            # ドーパミンとセロトニンによる出力調整
            attn_output = attn_output * dopamine_effect * serotonin_effect
            
            # アストロサイトの更新
            if self.astrocyte is not None:
                # アテンションをニューロン活性として解釈
                neural_activity = torch.reshape(attn_output[0], self.astrocyte.region_shape).detach().cpu().numpy()
                
                # アストロサイト状態の更新
                self.astrocyte.update(neural_activity)
                
                # アストロサイトの調節効果を取得
                astro_effects = self.astrocyte.get_modulatory_effect()
                
                # グルタミン酸・GABA取り込みの効果をアテンションに適用
                # （テンソルに変換して形状を合わせる）
                glutamate_uptake = torch.tensor(astro_effects['glutamate_uptake'], device=attn_output.device)
                synapse_mod = torch.tensor(astro_effects['synapse_modulation'], device=attn_output.device)
                
                # アストロサイトの効果を適用（次元を合わせる必要あり）
                glutamate_uptake = glutamate_uptake.view(*self.astrocyte.region_shape, 1).expand(-1, -1, attn_output.size(1))
                synapse_mod = synapse_mod.view(*self.astrocyte.region_shape, 1).expand(-1, -1, attn_output.size(1))
                
                # 次元を合わせる
                glutamate_uptake = glutamate_uptake.reshape(attn_output.shape)
                synapse_mod = synapse_mod.reshape(attn_output.shape)
                
                # アストロサイト効果の適用
                attn_output = attn_output * glutamate_uptake * synapse_mod
        
        if need_weights:
            return attn_output, attn_weights
        else:
            return attn_output


class BioKANBlock(nn.Module):
    """BioKANの基本構成ブロック"""
    
    def __init__(self, in_features, hidden_dim, out_features, 
                n_layers=2, activation='tanh', use_bias=True, neuromodulation=True):
        """
        初期化
        
        Args:
            in_features: 入力特徴量数
            hidden_dim: 隠れ層の次元
            out_features: 出力特徴量数
            n_layers: 層数
            activation: 活性化関数
            use_bias: バイアスを使用するかどうか
            neuromodulation: 神経調節を有効にするかどうか
        """
        super().__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.n_layers = n_layers
        
        # 三値活性化関数
        if activation == 'ternary':
            self.activation = TernaryActivationFunction()
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = getattr(torch, activation)
        
        # レイヤーを構築
        layers = []
        current_dim = in_features
        
        for i in range(n_layers - 1):
            layers.append(KANLinear(current_dim, hidden_dim, use_bias=use_bias))
            current_dim = hidden_dim
        
        layers.append(KANLinear(current_dim, out_features, use_bias=use_bias))
        
        self.layers = nn.ModuleList(layers)
        
        # 神経調節システム（オプション）
        self.neuromodulator = NeuromodulatorSystem() if neuromodulation else None
        
        # バイオロジカルアテンション（特徴間）
        self.feature_attention = BiologicalMultiHeadAttention(
            out_features, num_heads=4, neuromodulation=neuromodulation
        )
    
    def forward(self, x):
        """
        フォワードパス
        
        Args:
            x: 入力テンソル [batch_size, in_features]
            
        Returns:
            出力テンソル [batch_size, out_features]
        """
        # フィード前向き経路
        h = x
        hidden_states = [h]
        
        for i, layer in enumerate(self.layers):
            h = layer(h)
            
            # 中間層では活性化関数を適用
            if i < len(self.layers) - 1:
                h = self.activation(h)
                
            hidden_states.append(h)
        
        # 神経調節の影響を適用（もし有効なら）
        if self.neuromodulator is not None:
            # 活性化パターンに基づいて神経調節システムを更新
            h_abs = torch.abs(h)
            activity_level = h_abs.mean().item()
            sparsity = (h == 0).float().mean().item()
            
            # 神経伝達物質への刺激を計算
            stimuli = {
                'dopamine': 0.1 * activity_level - 0.05 * sparsity,  # 活動が高く、スパース性が低いとドーパミン放出
                'acetylcholine': 0.2 * (1 - sparsity),               # スパース性が低いとアセチルコリン放出
                'glutamate': 0.15 * activity_level,                  # 活動レベルに比例してグルタミン酸放出
                'gaba': 0.1 * sparsity                               # スパース性に比例してGABA放出
            }
            
            self.neuromodulator.update(stimuli=stimuli)
            
            # 神経伝達物質の状態を取得
            neuro_state = self.neuromodulator.get_state()
            
            # 出力の調整
            h = h * (1.0 + 0.2 * neuro_state['dopamine'])  # ドーパミンによる出力の増幅
            
            # ノルアドレナリンによる注意調整（閾値変更）
            attention_threshold = 0.5 - 0.3 * neuro_state['noradrenaline']
            h = torch.where(torch.abs(h) > attention_threshold, h, torch.zeros_like(h))
        
        # フィーチャーアテンション（出力の特徴間関係をモデル化）
        # アテンションを適用するために次元を追加して変換
        h_attn = h.unsqueeze(1)  # [batch_size, 1, out_features]
        
        # セルフアテンションとして適用
        h_attn, _ = self.feature_attention(h_attn, h_attn, h_attn)
        
        # 元の次元に戻す
        h = h_attn.squeeze(1)  # [batch_size, out_features]
        
        return h


class CorticalAttention(nn.Module):
    """皮質型アテンションメカニズム"""
    
    def __init__(self, embed_dim, num_regions=4, region_heads=2, dropout=0.1):
        """
        初期化
        
        Args:
            embed_dim: 埋め込み次元
            num_regions: 脳領域の数（前頭前皮質、頭頂葉、側頭葉など）
            region_heads: 各領域のアテンションヘッド数
            dropout: ドロップアウト確率
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_regions = num_regions
        self.region_heads = region_heads
        
        # 各「脳領域」のアテンション
        self.region_attentions = nn.ModuleList([
            BiologicalMultiHeadAttention(
                embed_dim // num_regions,
                region_heads,
                dropout=dropout
            )
            for _ in range(num_regions)
        ])
        
        # 領域間統合のためのアテンション
        self.integration_attention = BiologicalMultiHeadAttention(
            embed_dim, num_heads=num_regions, dropout=dropout
        )
        
        # 出力投影
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        フォワードパス
        
        Args:
            x: 入力テンソル [batch_size, seq_len, embed_dim]
            
        Returns:
            出力テンソル [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # 入力を「脳領域」に分割
        region_size = self.embed_dim // self.num_regions
        regions = x.view(batch_size, seq_len, self.num_regions, region_size)
        
        # 各領域で独立にアテンション計算
        region_outputs = []
        for i, attention in enumerate(self.region_attentions):
            region_input = regions[:, :, i, :]  # [batch_size, seq_len, region_size]
            
            # 各領域内でのセルフアテンション
            region_output, _ = attention(region_input, region_input, region_input)
            region_outputs.append(region_output)
        
        # 領域出力を連結
        concatenated = torch.cat(region_outputs, dim=-1)  # [batch_size, seq_len, embed_dim]
        
        # 領域間の情報統合（階層的注意）
        integrated, _ = self.integration_attention(concatenated, concatenated, concatenated)
        
        # 最終出力投影
        output = self.output_projection(integrated)
        output = self.dropout(output)
        
        return output


class HierarchicalMultiScaleAttention(nn.Module):
    """階層的マルチスケールアテンション"""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, num_scales=3):
        """
        初期化
        
        Args:
            embed_dim: 埋め込み次元
            num_heads: ヘッド数
            dropout: ドロップアウト確率
            num_scales: スケール数（時間・空間スケール）
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_scales = num_scales
        
        # マルチスケールアテンション
        # 各スケールは異なる範囲の情報に焦点を当てる
        self.scale_attentions = nn.ModuleList([
            BiologicalMultiHeadAttention(
                embed_dim,
                num_heads,
                dropout=dropout
            )
            for _ in range(num_scales)
        ])
        
        # スケール重み（学習可能）
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        # 出力投影
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        フォワードパス
        
        Args:
            x: 入力テンソル [batch_size, seq_len, embed_dim]
            mask: アテンションマスク
            
        Returns:
            出力テンソル [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # 各スケールでのアテンション計算
        scale_outputs = []
        
        for i, attention in enumerate(self.scale_attentions):
            # スケールに応じたアテンションマスクの調整
            scale_mask = mask
            
            if mask is not None and i > 0:
                # 大きなスケールでは、より広い範囲に注目できるようにマスクを調整
                # （実装例：スケールに応じて異なる数のトークンを参照可能にする）
                pass
            
            # アテンション計算
            scale_output, _ = attention(x, x, x, attn_mask=scale_mask)
            scale_outputs.append(scale_output)
        
        # スケール重みのソフトマックス正規化
        scale_weights = F.softmax(self.scale_weights, dim=0)
        
        # 重み付き集約
        output = torch.zeros_like(x)
        for i, scale_output in enumerate(scale_outputs):
            output += scale_weights[i] * scale_output
        
        # 最終出力投影
        output = self.output_projection(output)
        output = self.dropout(output)
        
        return output


class BioKANModel(nn.Module):
    """生物学的コルモゴロフ・アーノルド・ネットワークモデル"""
    
    def __init__(self, in_features, hidden_dim, num_classes, num_blocks=3, 
                attention_type='biological', dropout=0.1, neuromodulation=True):
        """
        初期化
        
        Args:
            in_features: 入力特徴量の数
            hidden_dim: 隠れ層の次元
            num_classes: 出力クラス数
            num_blocks: BioKANブロック数
            attention_type: アテンションの種類
                'biological': 標準の生物学的アテンション
                'cortical': 皮質型アテンション
                'hierarchical': 階層的マルチスケールアテンション
            dropout: ドロップアウト確率
            neuromodulation: 神経調節を有効にするかどうか
        """
        super().__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.neuromodulation = neuromodulation
        
        # ブロックの構築
        blocks = []
        block_in_dim = in_features
        
        for i in range(num_blocks):
            block_out_dim = hidden_dim
            
            blocks.append(
                BioKANBlock(
                    block_in_dim, 
                    hidden_dim, 
                    block_out_dim,
                    neuromodulation=neuromodulation
                )
            )
            
            block_in_dim = block_out_dim
        
        self.blocks = nn.ModuleList(blocks)
        
        # グローバルアテンションメカニズム
        if attention_type == 'cortical':
            self.global_attention = CorticalAttention(
                hidden_dim, num_regions=4, region_heads=2, dropout=dropout
            )
        elif attention_type == 'hierarchical':
            self.global_attention = HierarchicalMultiScaleAttention(
                hidden_dim, num_heads=8, dropout=dropout, num_scales=3
            )
        else:  # 'biological'
            self.global_attention = BiologicalMultiHeadAttention(
                hidden_dim, num_heads=8, dropout=dropout, neuromodulation=neuromodulation
            )
        
        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 神経伝達物質システム
        if neuromodulation:
            self.global_neuromodulator = NeuromodulatorSystem()
    
    def forward(self, x):
        """
        フォワードパス
        
        Args:
            x: 入力テンソル [batch_size, in_features] または [batch_size, seq_len, in_features]
            
        Returns:
            出力テンソル [batch_size, num_classes]
        """
        # 入力の形状を処理
        if x.dim() == 2:
            # [batch_size, in_features] -> [batch_size, 1, in_features]
            # （アテンションのための次元追加）
            is_sequence = False
            x = x.unsqueeze(1)
        else:
            is_sequence = True
        
        batch_size, seq_len, _ = x.size()
        
        # 各ブロックを通じて処理
        for block in self.blocks:
            # バッチとシーケンスを統合して処理
            x_flat = x.reshape(-1, x.size(-1))  # [batch_size*seq_len, feat_dim]
            
            # ブロック処理
            x_flat = block(x_flat)
            
            # 元の次元に戻す
            x = x_flat.view(batch_size, seq_len, -1)  # [batch_size, seq_len, feat_dim]
        
        # グローバルアテンションメカニズムの適用
        if isinstance(self.global_attention, BiologicalMultiHeadAttention):
            # マルチヘッドアテンションの場合
            x, _ = self.global_attention(x, x, x)
        else:
            # CorticalAttention または HierarchicalMultiScaleAttention の場合
            x = self.global_attention(x)
        
        # グローバル神経伝達物質の影響を適用
        if self.neuromodulation and hasattr(self, 'global_neuromodulator'):
            # シーケンス活動に基づいて神経調節を更新
            avg_activity = x.mean(dim=1).abs().mean(dim=1)  # [batch_size]
            max_activity, _ = x.abs().max(dim=1)  # [batch_size, feat_dim]
            max_activity = max_activity.mean(dim=1)  # [batch_size]
            
            # バッチの平均活動を使用
            avg_batch_activity = avg_activity.mean().item()
            max_batch_activity = max_activity.mean().item()
            
            # 神経伝達物質への刺激
            stimuli = {
                'dopamine': 0.1 * max_batch_activity,
                'acetylcholine': 0.2 * avg_batch_activity,
                'noradrenaline': 0.15 * max_batch_activity - 0.05 * avg_batch_activity,
                'serotonin': 0.1 * avg_batch_activity - 0.05 * max_batch_activity
            }
            
            self.global_neuromodulator.update(stimuli=stimuli)
            
            # 神経伝達物質の状態を取得
            neuro_state = self.global_neuromodulator.get_state()
            
            # 注意と覚醒の効果
            attention_factor = 1.0 + 0.3 * neuro_state['acetylcholine']  # アセチルコリンは注意力を高める
            arousal_factor = 1.0 + 0.2 * neuro_state['noradrenaline']    # ノルアドレナリンは覚醒を高める
            
            # 出力修正
            x = x * attention_factor * arousal_factor
        
        # 非シーケンスデータの場合はシーケンス次元を削除
        if not is_sequence:
            x = x.squeeze(1)
        else:
            # シーケンスの平均を取る
            x = x.mean(dim=1)
        
        # 分類
        logits = self.classifier(x)
        
        return logits


def create_biokan_classifier(in_features, hidden_dim=128, num_classes=10, 
                          num_blocks=3, attention_type='biological', 
                          dropout=0.1, neuromodulation=True):
    """
    BioKAN分類器を作成するヘルパー関数
    
    Args:
        in_features: 入力特徴量の数
        hidden_dim: 隠れ層の次元
        num_classes: 出力クラス数
        num_blocks: BioKANブロック数
        attention_type: アテンションの種類
            'biological', 'cortical', 'hierarchical'
        dropout: ドロップアウト確率
        neuromodulation: 神経調節を有効にするかどうか
        
    Returns:
        BioKANモデルのインスタンス
    """
    
    return BioKANModel(
        in_features=in_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_blocks=num_blocks,
        attention_type=attention_type,
        dropout=dropout,
        neuromodulation=neuromodulation
    ) 