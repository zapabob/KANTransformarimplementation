import torch
from torch.utils.data import DataLoader
from composer import Trainer
from composer.algorithms import GradientClipping, LayerFreezing
from composer.callbacks import LRMonitor, OptimizerMonitor
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from composer.schedulers import CosineAnnealingWithWarmupScheduler

from kan.core import KANTransformer
from kan.data import create_dataloaders

def create_dataloaders(batch_size: int = 32):
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

def main():
    # モデルの初期化
    model = KANTransformer(
        num_layers=6,
        dim=256,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        base_theta=0.5,
        k_dop=0.2,
        k_nor=0.15,
        k_sero=0.1,
        num_classes=10
    )
    
    # データローダーの作成
    train_loader, val_loader = create_dataloaders()
    
    # オプティマイザとスケジューラの設定
    optimizer = DecoupledAdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    scheduler = CosineAnnealingWithWarmupScheduler(
        t_warmup='1ep',
        t_max='10ep'
    )
    
    # Composerアルゴリズムの設定
    algorithms = [
        GradientClipping(clipping_type='norm', clipping_threshold=1.0),
        LayerFreezing(
            freeze_start='0.5ep',
            freeze_level=0.3
        )
    ]
    
    # コールバックの設定
    callbacks = [
        LRMonitor(),
        OptimizerMonitor()
    ]
    
    # WandBロガーの設定
    wandb_logger = WandBLogger(
        project='kan-transformer',
        name='experiment-1'
    )
    
    # Trainerの設定と実行
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        max_duration='10ep',
        optimizers=optimizer,
        schedulers=scheduler,
        algorithms=algorithms,
        callbacks=callbacks,
        loggers=[wandb_logger],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        eval_interval='1ep',
        save_folder='checkpoints',
        save_interval='1ep',
        save_num_checkpoints_to_keep=2,
        grad_accum=2,
        precision='amp'
    )
    
    # トレーニング実行
    trainer.fit()

if __name__ == '__main__':
    main() 