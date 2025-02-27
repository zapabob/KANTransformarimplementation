import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from .layers import TransformerBlock

class KANTransformer(nn.Module):
    def __init__(self,
                 num_layers: int = 6,
                 dim: int = 256,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 base_theta: float = 0.5,
                 k_dop: float = 0.2,
                 k_nor: float = 0.15,
                 k_sero: float = 0.1,
                 num_classes: int = 10):
        super().__init__()
        
        self.embedding = nn.Linear(dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, dim))  # 固定長100を仮定
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                base_theta=base_theta,
                k_dop=k_dop,
                k_nor=k_nor,
                k_sero=k_sero
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
        # XAI用の中間状態保存
        self.neuromod_states = []
        self.attention_maps = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # 位置エンコーディングの追加
        x = self.embedding(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # 中間状態の初期化
        self.neuromod_states = []
        self.attention_maps = []
        
        # Transformerブロックの処理
        for block in self.blocks:
            x = block(x)
            # 中間状態の保存
            self.neuromod_states.append(block.get_neuromod_state())
        
        # 分類ヘッド
        x = self.norm(x)
        x = x.mean(dim=1)  # グローバルプーリング
        x = self.head(x)
        
        return x
    
    def loss_fn(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(outputs, targets)
    
    def get_explanation(self, x: torch.Tensor) -> Dict[str, Any]:
        """XAI用の説明を生成"""
        # 推論実行
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            
        # 説明データの収集
        explanation = {
            'prediction': outputs.argmax(dim=-1),
            'confidence': torch.softmax(outputs, dim=-1).max(dim=-1)[0],
            'neuromod_states': self.neuromod_states,
            'attention_maps': self.attention_maps
        }
        
        return explanation
    
    def get_neuromod_visualization(self) -> Dict[str, torch.Tensor]:
        """神経調節状態の可視化用データを取得"""
        vis_data = {
            'dopamine': torch.stack([state['dopamine'] for state in self.neuromod_states]),
            'noradrenaline': torch.stack([state['noradrenaline'] for state in self.neuromod_states]),
            'serotonin': torch.stack([state['serotonin'] for state in self.neuromod_states])
        }
        return vis_data 