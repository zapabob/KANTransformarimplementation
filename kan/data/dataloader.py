import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

class KANDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data: shape [samples, sequence_length, features]
            labels: shape [samples]
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_dataloaders(
    batch_size: int = 32,
    train_size: int = 1000,
    val_size: int = 200,
    seq_length: int = 100,
    feature_dim: int = 256,
    num_workers: int = 4
):
    """データローダーを作成する関数
    
    Args:
        batch_size: バッチサイズ
        train_size: 訓練データのサンプル数
        val_size: 検証データのサンプル数
        seq_length: シーケンス長
        feature_dim: 特徴量の次元
        num_workers: データローダーのワーカー数
    """
    logger.info('データローダーを作成中...')
    
    # 訓練データの生成
    train_data = torch.randn(train_size, seq_length, feature_dim)
    train_labels = torch.randint(0, 10, (train_size,))
    
    # 検証データの生成
    val_data = torch.randn(val_size, seq_length, feature_dim)
    val_labels = torch.randint(0, 10, (val_size,))
    
    # データセットの作成
    train_dataset = KANDataset(train_data, train_labels)
    val_dataset = KANDataset(val_data, val_labels)
    
    # データローダーの作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f'訓練データ: {len(train_dataset)}サンプル')
    logger.info(f'検証データ: {len(val_dataset)}サンプル')
    
    return train_loader, val_loader 