# 拡張バイオミメティックKAN（神経調節機能強化版）

このリポジトリは、神経科学的に着想を得た拡張KAN（カーンインゴルドプレログ定理ネットワーク）モデルの実装です。従来のニューラルネットワークと比較して、このモデルは生物学的なニューロモジュレーターとグリア細胞の機能を模倣し、非同期的情報処理を実現しています。

## 主な特徴

### 1. 拡張ニューロモジュレーション
- **複数の神経伝達物質のシミュレーション**：
  - ドーパミン：報酬と動機付けの調整
  - ノルアドレナリン：注意と覚醒の制御
  - セロトニン：感情と学習率の調整
  - アセチルコリン：運動制御と記憶の強化
  - グルタミン酸：興奮性シグナルの伝達
  - GABA：抑制性シグナルの調整

### 2. グリア細胞機能の実装
- **アストロサイト**：
  - 長期コンテキスト記憶の保持
  - 神経活動への代謝サポート提供
  - 学習率の動的調整
  
- **ミクログリア**：
  - 自己修復メカニズム
  - 使用頻度の低いシナプス結合のプルーニング
  - ニューラルネットワークの健全性維持

### 3. 非同期的情報処理
- 異なる時間スケールで動作する神経回路のシミュレーション
- 時間依存的な三値活性化関数（-1, 0, 1）
- スパイキングニューラルネットワークの要素を統合

### 4. 説明可能AI（XAI）機能
- 内部状態の詳細な可視化
- 反実仮想シミュレーション（「もし～だったら」の分析）
- 意思決定プロセスの透明性

### 5. 運動制御機能
- genesisライブラリとの統合
- アセチルコリンによる制御される運動出力
- 階層的皮質構造の模倣

## 使用方法

### インストール

```bash
git clone https://github.com/zapabob/KANTransformarimplementation.git
cd KANTransformarimplementation
pip install -e .
```

または依存関係のみをインストールする場合：

```bash
pip install -r requirements.txt
```

### 非同期KANモデルのデモ実行

```bash
python examples/async_model_demo.py --layers 4 --dim 64 --time_steps 100
```

### パラメータ
- `--dim`: モデルの次元数
- `--layers`: レイヤー数
- `--heads`: アテンションヘッド数
- `--time_steps`: シミュレーション時間ステップ数
- `--output_dir`: 結果の出力先ディレクトリ
- `--motor_output_dim`: 運動出力の次元数
- `--seed`: 乱数シード

## コードの構成

- `kan/core/extended_neuromod.py`: 拡張された神経調節モジュールの実装
- `kan/core/genesis_integration.py`: genesisライブラリとの運動制御統合
- `kan/core/asynchronous_kan.py`: 非同期的な情報処理を行うKANモデル
- `examples/async_model_demo.py`: 非同期KANモデルのデモスクリプト

## 出力サンプル

デモスクリプトは以下のような可視化結果を生成します：

1. 神経調節物質の時間的変化
2. グリア細胞活動とシナプス結合強度の変化
3. 運動出力の時間的変化
4. 各種神経調節状態による反実仮想シミュレーション結果

## 今後の展望

1. さらなる生物学的妥当性の向上
2. リアルタイム運動制御システムへの応用
3. 長期記憶とエピソード記憶のモデリング
4. 感情状態を含む内部状態のモデリング
5. マルチモーダル入力への対応

## ライセンス

MIT License

## 引用

このプロジェクトを研究で使用する場合は、以下の形式で引用してください：

```bibtex
@software{KANTransformarimplementation2025,
  author = {Ryo Minegishi},
  title = {KAN Transformer: Neural-Modulated Ternary Activation Network},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/zapabob/KANTransformarimplementation}
}
```
