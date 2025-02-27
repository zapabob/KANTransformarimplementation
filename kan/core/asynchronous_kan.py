import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

from .layers import TernaryActivationFunction
from .extended_neuromod import ExtendedNeuromodulator, Astrocyte, Microglia, AsynchronousSpiking
from .genesis_integration import MotorCortexLayer, CorticalLayerStructure

class AsynchronousKANLayer(nn.Module):
    """
    非同期的情報処理を行うKAN層
    - 三値活性化関数に時間的要素（スパイキングタイミング）を導入
    - 神経細胞とグリア細胞の相互作用を模倣
    """
    def __init__(self, 
                dim: int, 
                num_heads: int = 8, 
                mlp_ratio: int = 4, 
                dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        # 基本的なTransformerブロック構造
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # 拡張された神経調節モジュール
        self.neuromodulator = ExtendedNeuromodulator()
        
        # グリア細胞モジュール
        self.astrocyte = Astrocyte(memory_size=1000)
        self.microglia = Microglia(memory_size=500)
        
        # 非同期スパイキングモジュール
        self.spiking_module = AsynchronousSpiking(num_neurons=dim)
        
        # 結合強度（ニューラルネットワークの重み）
        self.register_buffer('connection_strengths', torch.ones(dim, dim, dtype=torch.float32) * 0.5)
        
        # 時間ステップカウンター
        self.time_step = 0
        
    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        順伝播計算（非同期情報処理）
        x: 入力テンソル
        dt: 時間ステップサイズ
        """
        # 現在の時間ステップを更新
        self.time_step += 1
        
        # 自己注意機構
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        
        # 神経調節状態の取得
        neuromod_state = self.neuromodulator.get_state()
        
        # グリア細胞による変調の適用
        astro_context = self.astrocyte.get_context_modulation()
        metabolic_support = self.astrocyte.get_metabolic_support()
        
        # ミクログリアによる修復と最適化
        repair_factors = self.microglia.get_repair_factors()
        
        # MLPでの処理
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out * metabolic_support  # 代謝サポートによる変調
        
        # 結合強度の更新（ミクログリアのプルーニングに基づく）
        if self.time_step % 10 == 0:  # 10ステップごとに更新
            # 使用頻度の低い結合を弱める
            pruning_mask = (self.connection_strengths < repair_factors['pruning_threshold']).float()
            self.connection_strengths = self.connection_strengths * (1 - 0.1 * pruning_mask)
            
            # 頻繁に使用される結合を強化
            # テンソルの形状に合わせて活動マスクを作成
            activity = (torch.abs(x) > 0.5).float()
            # バッチとシーケンス次元の平均を取る
            activity_mean = activity.mean(dim=[0, 1])
            # 結合強度の形状に合わせて拡張
            activity_mask = activity_mean.unsqueeze(0).repeat(self.dim, 1)
            
            self.connection_strengths = self.connection_strengths + 0.01 * activity_mask
            
            # 結合強度を0～1の範囲に制限
            self.connection_strengths = torch.clamp(self.connection_strengths, 0, 1)
        
        # 神経調節物質に基づく三値活性化閾値の計算
        threshold = 0.5 + 0.1 * (neuromod_state['dopamine'] - neuromod_state['noradrenaline'] + 
                                neuromod_state['serotonin'] - neuromod_state['glutamate'] + 
                                neuromod_state['gaba'])
        
        # 非同期スパイキングによる三値活性化
        # バッチ次元を処理するために形状を調整
        batch_size, seq_len, dim = x.shape
        x_reshaped = x.reshape(-1, dim)  # バッチとシーケンスを結合
        
        # CPU上のnumpy配列に変換（1次元ずつ処理）
        spikes_list = []
        for i in range(x_reshaped.shape[0]):
            x_np = x_reshaped[i].detach().cpu().numpy().astype(np.float32)
            spike = self.spiking_module.update(x_np, neuromod_state, dt)
            spikes_list.append(spike)
        
        # スパイクを結合して元の形状に戻す
        spikes = np.stack(spikes_list, axis=0).reshape(batch_size, seq_len, dim)
        
        # スパイクを三値表現に変換 (-1, 0, 1)
        ternary_output = torch.tensor(spikes, device=x.device, dtype=torch.float32) * 2 - 1
        ternary_output = ternary_output * (torch.abs(x) > threshold).float()
        
        # 神経調節状態の更新（活動に基づく）
        activity = torch.abs(x).mean().item()
        sparsity = (torch.abs(x) < threshold).float().mean().item()
        stability = 1.0 - torch.std(x).item()
        motor_activity = torch.abs(torch.diff(x, dim=1)).mean().item() if x.size(1) > 1 else 0.0
        excitation_rate = (x > x.mean()).float().mean().item()
        
        self.neuromodulator.update(
            activity=activity,
            sparsity=sparsity,
            stability=stability,
            motor_activity=motor_activity,
            excitation_rate=excitation_rate,
            time_step=self.time_step
        )
        
        # グリア細胞の状態更新
        # バッチとシーケンスの次元を考慮して平均を取る
        x_mean = x.detach().cpu().numpy().mean(axis=(0, 1)).astype(np.float32)
        conn_np = self.connection_strengths.detach().cpu().numpy().astype(np.float32)
        
        self.astrocyte.update(x_mean, time_scale=dt*0.01)
        self.microglia.update(x_mean, conn_np)
        
        return ternary_output
    
    def get_state_representation(self) -> Dict[str, Any]:
        """モデルの内部状態表現を取得（説明可能性のため）"""
        state = {
            'neuromod': self.neuromodulator.get_state(),
            'astrocyte': self.astrocyte.get_state(),
            'microglia': self.microglia.get_state(),
            'spiking': self.spiking_module.get_state(),
            'connection_stats': {
                'mean': self.connection_strengths.mean().item(),
                'std': self.connection_strengths.std().item(),
                'min': self.connection_strengths.min().item(),
                'max': self.connection_strengths.max().item(),
            }
        }
        return state


class ExtendedKANTransformer(nn.Module):
    """
    拡張されたKANTransformerモデル
    - 非同期的情報処理
    - グリア細胞機能の統合
    - 階層的神経調節プロファイル
    - genesisライブラリとの運動制御機能連携
    """
    def __init__(self,
                 num_layers: int = 6,
                 dim: int = 256,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 num_classes: int = 10,
                 motor_output_dim: int = 30):  # 運動出力の次元
        super().__init__()
        
        # 基本エンベディング層
        self.embedding = nn.Linear(dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, dim))  # 固定長100を仮定
        
        # 非同期KAN層
        self.blocks = nn.ModuleList([
            AsynchronousKANLayer(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 大脳皮質層構造
        self.cortical_structure = CorticalLayerStructure(
            input_dim=dim,
            output_dim=dim,
            num_layers=6,
            hidden_dim=dim,
            dropout=dropout
        )
        
        # 運動皮質層
        self.motor_cortex = MotorCortexLayer(
            input_dim=dim,
            output_dim=motor_output_dim,
            hidden_dim=dim//2
        )
        
        # 出力層
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
        # 内部状態の記録
        self.layer_states = []
        self.attention_maps = []
        self.motor_outputs = []
        
        # サンプル運動パターンの登録
        sample_patterns = {
            'walk': [np.random.uniform(-0.5, 0.5, (10, 3)) for _ in range(20)],
            'jump': [np.random.uniform(-0.7, 0.7, (10, 3)) for _ in range(10)],
            'grasp': [np.random.uniform(-0.3, 0.3, (10, 3)) for _ in range(15)]
        }
        self.motor_cortex.load_patterns(sample_patterns)
    
    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 入力がタプルの場合は最初の要素を取得
        if isinstance(x, tuple):
            x = x[0]
            
        B = x.shape[0]
        
        # エンベディングと位置エンコーディング
        x = self.embedding(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # 内部状態の初期化
        self.layer_states = []
        self.attention_maps = []
        self.motor_outputs = []
        
        # 各層の処理（非同期情報処理）
        for i, block in enumerate(self.blocks):
            x = block(x)
            # 各層の状態を記録
            self.layer_states.append(block.get_state_representation())
        
        # 大脳皮質層構造による処理
        # 最後の層のグリア細胞から神経調節状態を取得
        neuromod_state = self.blocks[-1].neuromodulator.get_state()
        cortical_out, layer_outputs = self.cortical_structure(x, neuromod_state)
        
        # 運動出力の生成
        motor_output = self.motor_cortex(
            cortical_out.mean(dim=1),  # グローバルプーリング
            acetylcholine_level=neuromod_state['acetylcholine']
        )
        self.motor_outputs.append(motor_output)
        
        # 分類出力
        x = self.norm(cortical_out)
        x = x.mean(dim=1)  # グローバルプーリング
        logits = self.head(x)
        
        return logits, motor_output
    
    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, targets)
    
    def get_explanation(self, x: torch.Tensor) -> Dict[str, Any]:
        """XAI用の説明を生成"""
        # 推論実行
        self.eval()
        with torch.no_grad():
            logits, motor_output = self(x)
            
        # 説明データの収集
        explanation = {
            'prediction': logits.argmax(dim=-1),
            'confidence': torch.softmax(logits, dim=-1).max(dim=-1)[0],
            'layer_states': self.layer_states,
            'motor_output': motor_output,
            
            # 神経調節物質の状態をレイヤーごとに集計
            'neuromod_summary': {
                'dopamine': [state['neuromod']['dopamine'] for state in self.layer_states],
                'noradrenaline': [state['neuromod']['noradrenaline'] for state in self.layer_states],
                'serotonin': [state['neuromod']['serotonin'] for state in self.layer_states],
                'acetylcholine': [state['neuromod']['acetylcholine'] for state in self.layer_states],
                'glutamate': [state['neuromod']['glutamate'] for state in self.layer_states],
                'gaba': [state['neuromod']['gaba'] for state in self.layer_states]
            },
            
            # グリア細胞の状態（最後の層から）
            'glial_state': {
                'astrocyte': self.layer_states[-1]['astrocyte'],
                'microglia': self.layer_states[-1]['microglia']
            }
        }
        
        return explanation
    
    def generate_counterfactual(self, x: torch.Tensor, target_neuromod: Dict[str, float]) -> Dict[str, Any]:
        """反実仮想シミュレーション（「もし異なる神経調節状態だったら？」）"""
        self.eval()
        
        # 元の予測を保存
        with torch.no_grad():
            original_logits, original_motor = self(x)
            original_pred = original_logits.argmax(dim=-1)
        
        # 各層の神経調節状態を一時的に変更
        original_states = []
        for block in self.blocks:
            # 元の状態を保存
            original_state = {
                'dopamine': block.neuromodulator.dopamine,
                'noradrenaline': block.neuromodulator.noradrenaline,
                'serotonin': block.neuromodulator.serotonin,
                'acetylcholine': block.neuromodulator.acetylcholine,
                'glutamate': block.neuromodulator.glutamate,
                'gaba': block.neuromodulator.gaba
            }
            original_states.append(original_state)
            
            # 状態を一時的に変更
            for key, value in target_neuromod.items():
                if hasattr(block.neuromodulator, key):
                    setattr(block.neuromodulator, key, value)
        
        # 変更後の予測
        with torch.no_grad():
            cf_logits, cf_motor = self(x)
            cf_pred = cf_logits.argmax(dim=-1)
        
        # 元の状態に戻す
        for i, block in enumerate(self.blocks):
            for key, value in original_states[i].items():
                setattr(block.neuromodulator, key, value)
        
        # 結果を返す
        return {
            'original_prediction': original_pred,
            'counterfactual_prediction': cf_pred,
            'original_confidence': torch.softmax(original_logits, dim=-1).max(dim=-1)[0],
            'counterfactual_confidence': torch.softmax(cf_logits, dim=-1).max(dim=-1)[0],
            'prediction_changed': (original_pred != cf_pred).cpu().numpy(),
            'original_motor': original_motor,
            'counterfactual_motor': cf_motor,
            'target_neuromod': target_neuromod
        }
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """可視化用のデータを収集"""
        return {
            'layer_states': self.layer_states,
            'motor_outputs': self.motor_outputs,
            'neuromod_summary': {
                'dopamine': [state['neuromod']['dopamine'] for state in self.layer_states],
                'noradrenaline': [state['neuromod']['noradrenaline'] for state in self.layer_states],
                'serotonin': [state['neuromod']['serotonin'] for state in self.layer_states],
                'acetylcholine': [state['neuromod']['acetylcholine'] for state in self.layer_states],
                'glutamate': [state['neuromod']['glutamate'] for state in self.layer_states],
                'gaba': [state['neuromod']['gaba'] for state in self.layer_states]
            }
        } 