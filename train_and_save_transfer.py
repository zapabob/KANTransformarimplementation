"""
BioKANモデルの転移学習と保存のためのスクリプト
Fashion-MNISTデータセットで転移学習を行い、結果を保存します
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import time

# CUDA情報の表示
if torch.cuda.is_available():
    print(f"CUDA利用可能: {torch.cuda.get_device_name(0)}")
    print(f"CUDA バージョン: {torch.version.cuda}")
    print(f"PyTorch バージョン: {torch.__version__}")
else:
    print("CUDA利用不可: CPUで実行します")

try:
    # biokan_training.pyからモデルをインポート
    from biokan_training import EnhancedBioKANModel
    from biokan_transfer_learning import TransferBioKANModel, get_dataset, fine_tune_model
except ImportError as e:
    print(f"モジュールのインポートエラー: {e}")
    sys.exit(1)

def main():
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='BioKANモデルの転移学習と保存')
    parser.add_argument('--pretrained_model', type=str, default='biokan_trained_models/best_biokan_model.pth',
                        help='事前学習済みモデルのパス')
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                        choices=['mnist', 'cifar10', 'fashion_mnist'],
                        help='使用するデータセット')
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help='タスクの種類')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=5,
                        help='エポック数')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学習率')
    parser.add_argument('--freeze', action='store_true',
                        help='事前学習済み層を凍結するかどうか')
    parser.add_argument('--use_cuda', action='store_true',
                        help='利用可能な場合、CUDAを使用する')
    parser.add_argument('--output_dir', type=str, default='transfer_models',
                        help='モデル保存ディレクトリ')
    args = parser.parse_args()
    
    # デバイスの設定
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDAを使用: {torch.cuda.get_device_name(0)}")
        # CUDNNベンチマークモードを有効化
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print("CPUを使用")
    
    # 保存ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 事前学習済みモデルの読み込み
    print(f"事前学習済みモデルを読み込み中: {args.pretrained_model}")
    
    # モデルファイルの存在確認
    if not os.path.exists(args.pretrained_model):
        print(f"エラー: 事前学習済みモデルファイル '{args.pretrained_model}' が見つかりません")
        return
    
    try:
        # ベースモデルの読み込み
        base_model = EnhancedBioKANModel()
        base_model.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'))
        print(f"事前学習済みモデルを読み込みました")
        
        # 転移学習モデルの作成
        print(f"転移学習モデルを作成中...")
        # データセットごとのクラス数設定
        num_classes = 10  # MNISTとFashion-MNISTは10クラス
        if args.dataset == 'cifar10':
            num_classes = 10  # CIFAR-10も10クラス
            
        transfer_model = TransferBioKANModel(
            pretrained_model=base_model,
            task_type=args.task_type,
            num_classes=num_classes,
            output_dim=1,  # 回帰タスクの場合の出力次元
            freeze_layers=args.freeze
        )
        
        # モデルをデバイスに転送
        transfer_model = transfer_model.to(device)
        
        # 勾配計算を確実に有効化（フリーズしていてもoutput_layerの勾配は必要）
        for param in transfer_model.parameters():
            param.requires_grad_(True)
        
        # もし事前学習済み層を凍結する場合は、基本モデルのパラメータのみをフリーズ
        if args.freeze:
            for param in transfer_model.pretrained_model.parameters():
                param.requires_grad_(False)
        
        # モデル情報の表示
        total_params = sum(p.numel() for p in transfer_model.parameters())
        trainable_params = sum(p.numel() for p in transfer_model.parameters() if p.requires_grad)
        
        print(f"転移学習モデル構成:")
        print(f"  タスク種類: {args.task_type}")
        print(f"  データセット: {args.dataset}")
        print(f"  事前学習層凍結: {args.freeze}")
        print(f"  合計パラメータ数: {total_params}")
        print(f"  学習可能パラメータ数: {trainable_params}")
        print(f"  デバイス: {device}")
        
        # データセットの取得
        print(f"\nデータセット {args.dataset} を読み込み中...")
        train_dataset, test_dataset = get_dataset(args.dataset)
        
        # データセットの分割（訓練・検証）
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        # ジェネレータのシード固定
        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset = random_split(
            train_dataset, [train_size, val_size], 
            generator=generator
        )
        
        # データローダーの作成
        train_loader = DataLoader(
            train_subset, 
            batch_size=args.batch_size, 
            shuffle=True,
            pin_memory=device.type == 'cuda',
            num_workers=4 if device.type == 'cuda' else 0
        )
        val_loader = DataLoader(
            val_subset, 
            batch_size=args.batch_size,
            pin_memory=device.type == 'cuda',
            num_workers=4 if device.type == 'cuda' else 0
        )
        
        # 転移学習の実行
        print(f"\n転移学習を開始します...")
        start_time = time.time()
        
        history, trained_model = fine_tune_model(
            model=transfer_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            task_type=args.task_type
        )
        
        training_time = time.time() - start_time
        print(f"\n転移学習完了: {training_time:.2f}秒")
        
        # モデルの保存
        model_name = f"transfer_model_{args.dataset}_{args.task_type}.pth"
        model_path = os.path.join(args.output_dir, model_name)
        torch.save(trained_model.state_dict(), model_path)
        print(f"転移学習モデルを保存しました: {model_path}")
        
        # 学習曲線のプロット
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='訓練損失')
        plt.plot(history['val_loss'], label='検証損失')
        plt.xlabel('エポック')
        plt.ylabel('損失')
        plt.legend()
        plt.title('学習曲線 - 損失')
        
        if args.task_type == 'classification':
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='訓練精度')
            plt.plot(history['val_acc'], label='検証精度')
            plt.xlabel('エポック')
            plt.ylabel('精度')
            plt.legend()
            plt.title('学習曲線 - 精度')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'training_history_{args.dataset}.png'))
        print(f"学習曲線を保存しました: {os.path.join(args.output_dir, f'training_history_{args.dataset}.png')}")
        
        # 転移学習モデルのテスト
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size,
            pin_memory=device.type == 'cuda',
            num_workers=4 if device.type == 'cuda' else 0
        )
        
        # テスト評価
        print("\nテストデータでの評価...")
        trained_model.eval()
        test_loss = 0.0
        
        if args.task_type == 'classification':
            test_correct = 0
            test_total = 0
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # データの形状整形
                if inputs.dim() > 2:
                    inputs = inputs.view(inputs.size(0), -1)
                
                # 推論
                outputs = trained_model(inputs)
                loss = criterion(outputs, targets)
                
                # 損失更新
                test_loss += loss.item() * inputs.size(0)
                
                if args.task_type == 'classification':
                    # 分類タスクの評価
                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()
        
        # テスト結果の表示
        test_loss /= len(test_loader.dataset)
        
        if args.task_type == 'classification':
            test_acc = test_correct / test_total
            print(f"テスト損失: {test_loss:.4f}")
            print(f"テスト精度: {test_acc:.4f}")
        else:
            print(f"テスト損失 (MSE): {test_loss:.4f}")
        
        print(f"\n転移学習が完了しました。モデルは {model_path} に保存されています。")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 