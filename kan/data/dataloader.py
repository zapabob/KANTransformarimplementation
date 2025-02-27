import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple

def create_dataloaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """データローダーを作成する関数

    Args:
        batch_size (int, optional): バッチサイズ. デフォルトは32.

    Returns:
        Tuple[DataLoader, DataLoader]: 訓練用と検証用のデータローダー
    """
    # ここでは例としてダミーデータを生成
    # 実際のアプリケーションでは適切なデータセットを使用
    train_data = torch.randn(1000, 100, 256)  # [samples, sequence_length, features]
    train_labels = torch.randint(0, 10, (1000,))
    val_data = torch.randn(200, 100, 256)
    val_labels = torch.randint(0, 10, (200,))
    
    train_loader = DataLoader(
        list(zip(train_data, train_labels)),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        list(zip(val_data, val_labels)),
        batch_size=batch_size
    )
    return train_loader, val_loader 