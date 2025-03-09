"""
MNISTデータセットでハイパーパラメータ最適化を行い、最適なパラメータで転移学習を行うスクリプト
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import json
import os
import argparse
from tqdm.auto import tqdm
import time

# 自作モジュールのインポート
from biokan_training import EnhancedBioKANModel
from biokan_transfer_learning import (
    TransferBioKANModel, 
    get_dataset, 
    optimize_hyperparameters, 
    fine_tune_model, 
    evaluate_model,
    visualize_results
)

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

# CUDA情報の表示（接続時）
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    print(f"CUDA バージョン: {cuda_version}")
    
    # CUDA 12の互換性チェック
    if cuda_version.startswith('12.'):
        print("CUDA 12が検出されました。最適化された機能を使用します。")
        # CUDA 12特有の最適化設定
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # 詳細なGPU情報
        current_device = torch.cuda.current_device()
        print(f"現在使用中のGPU: {torch.cuda.get_device_name(current_device)}")
        print(f"GPU メモリ合計: {torch.cuda.get_device_properties(current_device).total_memory / 1024**3:.2f} GB")
        print(f"GPU メモリ使用量: {torch.cuda.memory_allocated(current_device) / 1024**3:.2f} GB")
        print(f"GPU キャッシュ: {torch.cuda.memory_reserved(current_device) / 1024**3:.2f} GB")
    else:
        print(f"注意: CUDA {cuda_version}が検出されました。CUDA 12向けの最適化は利用できません。")
    
    # GPU情報
    device_count = torch.cuda.device_count()
    print(f"利用可能なGPUデバイス数: {device_count}")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {device_name}")
else:
    print("警告: GPUが検出されませんでした。CPUで実行されます（処理速度が大幅に低下します）")

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='MNISTのハイパーパラメータ最適化と転移学習')
    parser.add_argument('--pretrained_model', type=str, default='biokan_trained_models/best_biokan_model.pth',
                        help='事前学習済みモデルのパス')
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help='転移学習のタスク種類')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='データローダーのバッチサイズ')
    parser.add_argument('--n_trials', type=int, default=10,
                        help='ハイパーパラメータ最適化の試行回数')
    parser.add_argument('--optimize_only', action='store_true',
                        help='最適化のみ行い、転移学習は行わない')
    parser.add_argument('--train_only', action='store_true',
                        help='最適化を行わず、既存のパラメータで転移学習のみ行う')
    parser.add_argument('--params_file', type=str, default=None,
                        help='既存のパラメータファイル（--train_onlyの場合に使用）')
    
    args = parser.parse_args()
    
    # 1. 事前学習済みモデルのロード
    print("\n事前学習済みモデルをロード中...")
    try:
        if os.path.exists(args.pretrained_model):
            # オリジナルのモデル構造を作成
            base_model = EnhancedBioKANModel()
            # 事前学習済みの重みをロード
            base_model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
            base_model = base_model.to(device)
            print(f"事前学習済みモデルを読み込みました: {args.pretrained_model}")
        else:
            raise FileNotFoundError(f"モデルファイルが見つかりません: {args.pretrained_model}")
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        return
    
    # 2. MNISTデータセットの取得
    print("\nMNISTデータセットを準備中...")
    train_dataset, test_dataset = get_dataset('mnist')
    
    # 3. データセットの分割（訓練・検証・テスト）
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # 4. データローダーの作成
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
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        pin_memory=device.type == 'cuda',
        num_workers=4 if device.type == 'cuda' else 0
    )
    
    print(f"データセットの準備が完了しました")
    print(f"  訓練データ: {len(train_subset)}サンプル")
    print(f"  検証データ: {len(val_subset)}サンプル")
    print(f"  テストデータ: {len(test_dataset)}サンプル")
    
    best_params = None
    
    # 5. ハイパーパラメータの最適化または既存パラメータの読み込み
    if args.train_only and args.params_file:
        print(f"\n既存のパラメータファイルを読み込み中: {args.params_file}")
        try:
            with open(args.params_file, 'r') as f:
                best_params = json.load(f)
            print("パラメータの読み込みが完了しました")
            print("\n使用するハイパーパラメータ:")
            for param_name, param_value in best_params.items():
                print(f"  {param_name}: {param_value}")
        except Exception as e:
            print(f"パラメータファイルの読み込みに失敗しました: {e}")
            return
    elif not args.train_only:
        print(f"\nハイパーパラメータの最適化を開始します (試行回数: {args.n_trials})...")
        best_params, study = optimize_hyperparameters(
            base_model=base_model,
            train_loader=train_loader,
            val_loader=val_loader,
            task_type=args.task_type,
            n_trials=args.n_trials,
            save_best_params=True
        )
    
    # 最適化のみの場合はここで終了
    if args.optimize_only:
        print("\nハイパーパラメータ最適化が完了しました。")
        return
    
    # 6. 最適なパラメータで転移学習を実行
    if best_params is not None:
        print("\n最適なパラメータで転移学習を開始します...")
        
        # バッチサイズの調整（データローダーを再作成）
        if 'batch_size' in best_params:
            batch_size = best_params['batch_size']
            train_loader = DataLoader(
                train_subset, 
                batch_size=batch_size, 
                shuffle=True,
                pin_memory=device.type == 'cuda',
                num_workers=4 if device.type == 'cuda' else 0
            )
            val_loader = DataLoader(
                val_subset, 
                batch_size=batch_size,
                pin_memory=device.type == 'cuda',
                num_workers=4 if device.type == 'cuda' else 0
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size,
                pin_memory=device.type == 'cuda',
                num_workers=4 if device.type == 'cuda' else 0
            )
        
        # モデルの作成
        transfer_model = TransferBioKANModel(
            pretrained_model=base_model,
            task_type=args.task_type,
            num_classes=10 if args.task_type == 'classification' else 1,
            freeze_layers=best_params.get('freeze_layers', True)
        )
        
        # Dropout率の更新
        for module in transfer_model.modules():
            if isinstance(module, nn.Dropout):
                module.p = best_params.get('dropout', 0.2)
        
        transfer_model = transfer_model.to(device)
        
        # モデルのファインチューニング
        epochs = best_params.get('epochs', 10)
        learning_rate = best_params.get('learning_rate', 0.001)
        
        print(f"モデルのトレーニングを開始します...")
        print(f"  エポック数: {epochs}")
        print(f"  学習率: {learning_rate}")
        print(f"  凍結層: {best_params.get('freeze_layers', True)}")
        print(f"  Dropout率: {best_params.get('dropout', 0.2)}")
        
        start_time = time.time()
        
        history, trained_model = fine_tune_model(
            model=transfer_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=learning_rate,
            task_type=args.task_type
        )
        
        training_time = time.time() - start_time
        print(f"\nトレーニングが完了しました (所要時間: {training_time:.2f}秒)")
        
        # 7. 最終評価
        print("\nテストデータで評価中...")
        test_results = evaluate_model(
            model=trained_model,
            test_loader=test_loader,
            task_type=args.task_type
        )
        
        if args.task_type == 'classification':
            test_accuracy = test_results['accuracy']
            test_loss = test_results['loss']
            print(f"\nテスト精度: {test_accuracy*100:.2f}%")
            print(f"テスト損失: {test_loss:.4f}")
            
            # 混同行列の可視化
            visualize_results(
                targets=test_results['true_labels'],
                predictions=test_results['predictions'],
                task_type=args.task_type
            )
        else:
            test_mse = test_results['mse']
            test_r2 = test_results['r2']
            print(f"\nテストMSE: {test_mse:.4f}")
            print(f"テストR2スコア: {test_r2:.4f}")
        
        # 8. モデルの保存
        save_path = f'optimized_mnist_{args.task_type}_model.pth'
        torch.save(trained_model.state_dict(), save_path)
        print(f"\n最適化されたモデルを保存しました: {save_path}")
        
        # 9. 結果のまとめを保存
        results_summary = {
            'hyperparameters': best_params,
            'training_time': training_time,
            'test_results': test_results,
            'training_history': {k: [float(x) for x in v] for k, v in history.items()}
        }
        
        with open(f'optimized_mnist_{args.task_type}_results.json', 'w') as f:
            json.dump(results_summary, f, indent=4)
        
        print("\nMNIST転移学習の最適化と実行が完了しました！")
    else:
        print("エラー: 有効なパラメータがありません。転移学習を実行できません。")

if __name__ == "__main__":
    main() 