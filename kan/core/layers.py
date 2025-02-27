import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from composer.models import ComposerModel
from typing import Dict, Optional, Tuple

class TernaryActivationFunction(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(input, torch.tensor(threshold))
        return torch.where(input > threshold, torch.ones_like(input),
                         torch.where(input < -threshold, -torch.ones_like(input),
                                   torch.zeros_like(input)))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        input, threshold = ctx.saved_tensors
        grad_input = grad_output.clone() * (input.abs() < threshold).float()
        return grad_input, None

class NeuromodulatedAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # 神経調節用の投影層
        self.neuromod_proj = nn.Linear(3, dim)  # 3は神経調節物質の数
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor,
                neuromod_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, L, D = x.shape
        
        # 神経調節状態をテンソルに変換
        neuromod_tensor = torch.stack([
            neuromod_state['dopamine'],
            neuromod_state['noradrenaline'],
            neuromod_state['serotonin']
        ], dim=-1)  # [B, 3]
        
        # 神経調節による注意の変調
        neuromod_signal = self.neuromod_proj(neuromod_tensor)  # [B, D]
        neuromod_signal = neuromod_signal.unsqueeze(1)  # [B, 1, D]

        # 通常の自己注意処理
        q = self.q_proj(x + neuromod_signal)
        k = self.k_proj(x + neuromod_signal)
        v = self.v_proj(x)

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)

        return out

class KANLayer(nn.Module):
    def __init__(self, 
                 dim: int,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 base_theta: float = 0.5,
                 k_dop: float = 0.2,
                 k_nor: float = 0.15,
                 k_sero: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NeuromodulatedAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # 神経調節パラメータ
        self.base_theta = base_theta
        self.k_dop = k_dop
        self.k_nor = k_nor
        self.k_sero = k_sero
        
        # 状態の初期化
        self.register_buffer('dopamine', torch.zeros(1))
        self.register_buffer('noradrenaline', torch.zeros(1))
        self.register_buffer('serotonin', torch.zeros(1))

    def get_neuromod_state(self) -> Dict[str, torch.Tensor]:
        return {
            'dopamine': self.dopamine,
            'noradrenaline': self.noradrenaline,
            'serotonin': self.serotonin
        }

    def update_neuromod_state(self, output: torch.Tensor):
        # 出力の統計に基づいて神経調節状態を更新
        activity = output.abs().mean()
        sparsity = (output == 0).float().mean()
        
        # ドーパミン：高い活性で増加
        self.dopamine.data = torch.clamp(
            self.dopamine * 0.95 + 0.05 * activity, 0, 1)
        
        # ノルアドレナリン：スパース性で増加
        self.noradrenaline.data = torch.clamp(
            self.noradrenaline * 0.95 + 0.05 * sparsity, 0, 1)
        
        # セロトニン：安定性（活性の変動の少なさ）で増加
        stability = 1 - torch.std(output)
        self.serotonin.data = torch.clamp(
            self.serotonin * 0.95 + 0.05 * stability, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自己注意処理
        attn_out = self.attn(self.norm1(x), self.get_neuromod_state())
        x = x + attn_out
        
        # MLP処理
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        
        # 三値活性化
        threshold = (self.base_theta + 
                    self.k_dop * self.dopamine -
                    self.k_nor * self.noradrenaline +
                    self.k_sero * self.serotonin)
        x = TernaryActivationFunction.apply(x, threshold)
        
        # 神経調節状態の更新
        self.update_neuromod_state(x)
        
        return x 