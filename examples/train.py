import logging
import torch
import argparse
from kan.core import KANTransformer, ExtendedKANTransformer
from kan.data import create_dataloaders
from kan.training import Trainer
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import genesis
def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='KANモデルトレーニングスクリプト')
    parser.add_argument('--model', type=str, default='kan', choices=['kan', 'extended'],
                        help='使用するモデル: kan (標準) or extended (拡張版)')
    parser.add_argument('--epochs', type=int, default=10, help='トレーニングエポック数')
    parser.add_argument('--batch_size', type=int, default=32, help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=1e-4, help='学習率')
    parser.add_argument('--num_layers', type=int, default=6, help='モデル層数')
    parser.add_argument('--dim', type=int, default=256, help='モデル次元数')
    parser.add_argument('--num_heads', type=int, default=8, help='アテンションヘッド数')
    args = parser.parse_args()

    # ロギングの設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info('トレーニングを開始します')
    logger.info(f'モデルタイプ: {args.model}')
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用デバイス: {device}')
    
    # モデルの初期化
    if args.model == 'kan':
        logger.info('標準KANモデルを使用します')
        model = KANTransformer(
            num_layers=args.num_layers,
            dim=args.dim,
            num_heads=args.num_heads,
            mlp_ratio=4,
            dropout=0.1,
            base_theta=0.5,
            k_dop=0.2,
            k_nor=0.15,
            k_sero=0.1,
            num_classes=10
        ).to(device)
    else:
        logger.info('拡張KANモデル（生体模倣型）を使用します')
        model = ExtendedKANTransformer(
            num_layers=args.num_layers,
            dim=args.dim,
            num_heads=args.num_heads,
            mlp_ratio=4,
            dropout=0.1,
            num_classes=10,
            motor_output_dim=30
        ).to(device)
    
    # データローダーの作成
    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        feature_dim=args.dim
    )
    
    # オプティマイザとスケジューラの設定
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs - 1  # エポック数 - 1エポックのウォームアップ
    )
    
    # トレーナーの初期化と実行
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs
    )
    
    trainer.train()

if __name__ == '__main__':
    main() 