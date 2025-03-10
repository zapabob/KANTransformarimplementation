"""
Fashion MNISTデータセットを用いたBioKANモデルの転移学習
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

from cuda_info_manager import print_cuda_info, get_device, setup_japanese_fonts
from biokan_transfer_learning import TransferBioKANModel
from biokan_training import EnhancedBioKANModel

# 日本語フォントの設定
setup_japanese_fonts(verbose=False)

# デバイスの設定
device = get_device()
print_cuda_info(verbose=True)

def train_fashion_mnist():
    """Fashion MNISTデータセットを用いた転移学習を実行"""
    
    # データセットの準備
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Fashion MNISTデータセットのロード
    train_dataset = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=transform
    )
    
    # データローダーの設定
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # クラス名の定義
    class_names = ['Tシャツ/トップ', 'ズボン', 'プルオーバー', 'ドレス', 'コート',
                  'サンダル', 'シャツ', 'スニーカー', 'バッグ', 'アンクルブーツ']
    
    # 事前学習済みモデルのパス
    pretrained_path = 'optimized_mnist_classification_model.pth'
    
    # 事前学習済みモデルの読み込み
    state_dict = torch.load(pretrained_path, map_location=device)
    
    # state_dictのキーから'pretrained_model.'を削除
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('pretrained_model.'):
            new_key = key.replace('pretrained_model.', '')
            new_state_dict[new_key] = value
    
    # ベースモデルの作成と重みの読み込み
    base_model = EnhancedBioKANModel()
    base_model.load_state_dict(new_state_dict)
    
    # 転移学習モデルの作成
    model = TransferBioKANModel(
        pretrained_model=base_model,
        task_type='classification',
        num_classes=10,
        freeze_layers=True
    )
    model = model.to(device)
    
    # 損失関数とオプティマイザーの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 訓練ループ
    num_epochs = 10
    best_accuracy = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    for epoch in range(num_epochs):
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'エポック {epoch+1}/{num_epochs} [訓練]')
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.4f}'
            })
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # テストフェーズ
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f'エポック {epoch+1}/{num_epochs} [テスト]')
            for inputs, targets in test_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
                test_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{test_correct/test_total:.4f}'
                })
        
        test_loss = test_loss / len(test_loader.dataset)
        test_acc = test_correct / test_total
        
        # 学習率の調整
        scheduler.step(test_loss)
        
        # 履歴の更新
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f'\nエポック {epoch+1}/{num_epochs}:')
        print(f'訓練損失: {train_loss:.4f}, 訓練精度: {train_acc:.4f}')
        print(f'テスト損失: {test_loss:.4f}, テスト精度: {test_acc:.4f}')
        
        # 最良モデルの保存
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'transfer_model_fashion_mnist_classification.pth')
            print(f'最良モデルを保存しました（精度: {test_acc:.4f}）')
    
    # 学習曲線の描画
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='訓練')
    plt.plot(history['test_loss'], label='テスト')
    plt.title('損失の推移')
    plt.xlabel('エポック')
    plt.ylabel('損失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='訓練')
    plt.plot(history['test_acc'], label='テスト')
    plt.title('精度の推移')
    plt.xlabel('エポック')
    plt.ylabel('精度')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_fashion_mnist.png')
    plt.close()
    
    # 結果の保存
    results = {
        'best_accuracy': best_accuracy,
        'final_train_loss': history['train_loss'][-1],
        'final_train_acc': history['train_acc'][-1],
        'final_test_loss': history['test_loss'][-1],
        'final_test_acc': history['test_acc'][-1],
        'class_names': class_names
    }
    
    with open('fashion_mnist_transfer_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print('\n転移学習が完了しました。')
    print(f'最終テスト精度: {results["final_test_acc"]:.4f}')
    print(f'最良テスト精度: {best_accuracy:.4f}')

if __name__ == "__main__":
    train_fashion_mnist() 