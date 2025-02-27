# KAN Transformer

神経調節機能付き三値活性化ネットワーク（KAN: Kinetic Activation Network）の PyTorch Composer 実装

## 概要

このプロジェクトは、説明可能AIとヒューマノイドロボットの認知アーキテクチャを目指した新しいニューラルネットワークモデルを実装しています。

### 主な特徴

- 三値活性化関数（-1, 0, 1）による離散的な情報処理
- 神経調節機能（ドーパミン、ノルアドレナリン、セロトニン）による動的な振る舞いの制御
- Transformerアーキテクチャとの統合による高度な文脈処理
- 説明可能性（XAI）機能の組み込み

## インストール

```bash
git clone https://github.com/yourusername/kan-transformer.git
cd kan-transformer
pip install -e .
```

## 使用方法

### トレーニング

```bash
python examples/train.py
```

### 可視化

```bash
python examples/visualization_demo.py
```

## プロジェクト構造

```
kan_project/
├── README.md
├── requirements.txt
├── setup.py
├── kan/
│   ├── core/          # モデルのコア実装
│   ├── utils/         # ユーティリティ関数
│   └── data/          # データ処理
├── examples/          # 使用例
└── tests/            # テストコード
```

## 神経調節機能

- **ドーパミン**: 報酬系の模倣、高い活性で増加
- **ノルアドレナリン**: 注意系の模倣、スパース性で増加
- **セロトニン**: 情動系の模倣、安定性で増加

## 説明可能性

- 神経調節状態の可視化
- 注意マップの分析
- 意思決定経路の追跡

## 開発者向け情報

### テストの実行

```bash
pytest tests/
```

### コードスタイル

```bash
black kan/
flake8 kan/
```

## ライセンス

MIT License

## 引用

このプロジェクトを研究で使用する場合は、以下の形式で引用してください：

```bibtex
@software{kan_transformer2024,
  author = {Your Name},
  title = {KAN Transformer: Neural-Modulated Ternary Activation Network},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/kan-transformer}
}
``` 