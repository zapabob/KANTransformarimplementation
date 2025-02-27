import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import os
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.font_manager as fm
import sys

# 日本語フォントの設定
if sys.platform.startswith('win'):
    # Windowsの場合
    fonts = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'Yu Mincho']
    for font in fonts:
        try:
            plt.rcParams['font.family'] = font
            break
        except:
            continue
    
    # Windowsフォントパスを明示的に追加
    font_dirs = [
        'C:/Windows/Fonts',
        os.path.join(os.environ['LOCALAPPDATA'], 'Microsoft/Windows/Fonts')
    ]
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            font_files = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.endswith(('.ttf', '.ttc', '.otf'))]
            for font_file in font_files:
                try:
                    fm.fontManager.addfont(font_file)
                except:
                    pass
    
elif sys.platform.startswith('darwin'):
    # macOSの場合
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    # Linuxの場合
    fonts = fm.findSystemFonts()
    jp_fonts = [f for f in fonts if any(name in f.lower() for name in ['gothic', 'noto', 'meiryo', 'ipaexg'])]
    if jp_fonts:
        plt.rcParams['font.family'] = fm.FontProperties(fname=jp_fonts[0]).get_name()
    else:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Takao']

plt.rcParams['axes.unicode_minus'] = False  # マイナス記号を正しく表示

from kan.core.asynchronous_kan import ExtendedKANTransformer, AsynchronousKANLayer
from kan.core.extended_neuromod import ExtendedNeuromodulator

# コマンドライン引数のパース
parser = argparse.ArgumentParser(description='KANモデルの非同期処理と神経調節デモ')
parser.add_argument('--dim', type=int, default=64, help='モデルの次元数')
parser.add_argument('--layers', type=int, default=4, help='レイヤー数')
parser.add_argument('--heads', type=int, default=4, help='アテンションヘッド数')
parser.add_argument('--time_steps', type=int, default=100, help='シミュレーション時間ステップ数')
parser.add_argument('--output_dir', type=str, default='async_demo_results', help='出力ディレクトリ')
parser.add_argument('--motor_output_dim', type=int, default=10, help='運動出力の次元数')
parser.add_argument('--seed', type=int, default=42, help='乱数シード')
args = parser.parse_args()

# 出力ディレクトリの作成
os.makedirs(args.output_dir, exist_ok=True)

# シードの設定
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

# モデルの作成
model = ExtendedKANTransformer(
    num_layers=args.layers,
    dim=args.dim,
    num_heads=args.heads,
    mlp_ratio=4,
    dropout=0.1,
    num_classes=10,
    motor_output_dim=args.motor_output_dim
)
model = model.to(device)
print(f"モデル構造: {model}")

# 入力データの生成（ダミーデータ）
def generate_input(batch_size=1, seq_len=100, dim=64):
    x = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)
    y = torch.randint(0, 10, (batch_size,), dtype=torch.long)
    return x.to(device), y.to(device)

# 非同期モデルの動作をシミュレーション
def run_simulation(model, time_steps=100):
    results = {
        'neuromod_states': [],
        'layer_activations': [],
        'motor_outputs': [],
        'predictions': []
    }
    
    # 同じ入力を使い続ける
    x, y = generate_input(batch_size=1, seq_len=100, dim=args.dim)
    
    # モデルのデータ型を確認して統一
    model = model.to(torch.float32)
    
    # 時間ステップごとにモデルを実行
    for t in tqdm(range(time_steps), desc="シミュレーション実行中"):
        # 時間ステップを小さくして、より微細な変化を作る
        dt = 0.1 + 0.1 * np.sin(t / 10)  # 時間幅を変動させる
        
        # 各層の神経調節状態に摂動を加える（環境や内部状態の変化を模倣）
        if t % 10 == 0:
            perturbation = {
                'dopamine': 0.1 * np.sin(t / 25),
                'noradrenaline': 0.1 * np.cos(t / 20),
                'serotonin': 0.05 * np.sin(t / 30),
                'acetylcholine': 0.15 * np.cos(t / 15) if t > 50 else 0,
                'glutamate': 0.1 * np.sin(t / 10) if t % 5 == 0 else 0,
                'gaba': 0.05 * np.cos(t / 5) if t % 3 == 0 else 0
            }
            
            for layer in model.blocks:
                for key, value in perturbation.items():
                    if hasattr(layer.neuromodulator, key):
                        current = getattr(layer.neuromodulator, key)
                        setattr(layer.neuromodulator, key, 
                                max(0, min(1, current + value)))  # 0～1の範囲に制限
        
        # 変更された非同期的なdt値でモデルを実行
        for i, block in enumerate(model.blocks):
            # 各層で時間幅を少しずつ変える（非同期性）
            layer_dt = dt * (0.9 + 0.2 * np.random.random())
            
            if i == 0:
                prev_x = x
                x = block(x, dt=layer_dt)
            else:
                prev_x = x
                x = block(x, dt=layer_dt)
        
        # 最終的なモデル出力を取得
        model.eval()
        with torch.no_grad():
            logits, motor_output = model((prev_x, None))
            prediction = logits.argmax(dim=-1)
            
        # 結果を記録
        layer_states = []
        for layer in model.blocks:
            layer_states.append(layer.get_state_representation())
        
        results['neuromod_states'].append({
            't': t, 
            'dt': dt,
            'states': [
                {k: layer['neuromod'][k] for k in layer['neuromod']} 
                for layer in layer_states
            ]
        })
        
        results['layer_activations'].append({
            't': t,
            'activations': [
                {
                    'connection_mean': layer['connection_stats']['mean'],
                    'connection_std': layer['connection_stats']['std'],
                    'astrocyte_activity': layer['astrocyte']['activity'],
                    'microglia_activity': layer['microglia']['repair_rate']
                }
                for layer in layer_states
            ]
        })
        
        results['motor_outputs'].append({
            't': t,
            'output': motor_output.cpu().numpy().tolist()
        })
        
        results['predictions'].append({
            't': t,
            'prediction': prediction.item(),
            'confidence': torch.softmax(logits, dim=-1).max().item()
        })
    
    return results

# 反実仮想シミュレーション実行
def run_counterfactual_simulations(model, base_input):
    counterfactuals = []
    
    # 様々な神経調節状態に対する反実仮想
    neuromod_variations = [
        {'name': '高ドーパミン', 'dopamine': 0.9, 'noradrenaline': 0.5, 'serotonin': 0.5},
        {'name': '低ドーパミン', 'dopamine': 0.1, 'noradrenaline': 0.5, 'serotonin': 0.5},
        {'name': '高セロトニン', 'dopamine': 0.5, 'noradrenaline': 0.5, 'serotonin': 0.9},
        {'name': '低セロトニン', 'dopamine': 0.5, 'noradrenaline': 0.5, 'serotonin': 0.1},
        {'name': '高アセチルコリン', 'acetylcholine': 0.9, 'glutamate': 0.5, 'gaba': 0.5},
        {'name': '低アセチルコリン', 'acetylcholine': 0.1, 'glutamate': 0.5, 'gaba': 0.5},
        {'name': '高グルタミン酸', 'glutamate': 0.9, 'gaba': 0.3, 'acetylcholine': 0.5},
        {'name': '高GABA', 'glutamate': 0.3, 'gaba': 0.9, 'acetylcholine': 0.5},
    ]
    
    for variation in neuromod_variations:
        cf_result = model.generate_counterfactual(base_input, variation)
        cf_result['variation_name'] = variation['name']
        counterfactuals.append(cf_result)
    
    return counterfactuals

# ビジュアライゼーション関数
def visualize_simulation_results(results, output_dir):
    # フォント設定を再確認
    if sys.platform.startswith('win'):
        plt.rcParams['font.family'] = 'MS Gothic'
    
    # 1. 神経調節状態の時間的変化
    plt.figure(figsize=(15, 10))
    neuromod_names = ['dopamine', 'noradrenaline', 'serotonin', 
                      'acetylcholine', 'glutamate', 'gaba']
    neuromod_labels = ['ドーパミン', 'ノルアドレナリン', 'セロトニン', 
                      'アセチルコリン', 'グルタミン酸', 'GABA']
    
    for layer_idx in range(len(results['neuromod_states'][0]['states'])):
        plt.subplot(len(results['neuromod_states'][0]['states']), 1, layer_idx+1)
        for i, neuromod in enumerate(neuromod_names):
            values = [state['states'][layer_idx][neuromod] for state in results['neuromod_states']]
            plt.plot(range(len(values)), values, label=neuromod_labels[i])
        plt.title(f'レイヤー {layer_idx+1} の神経調節状態')
        plt.ylabel('活性度')
        plt.xlabel('時間ステップ')
        plt.legend(prop={'family': 'MS Gothic'})
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neuromodulator_states.png'))
    
    # 2. 運動出力の時間的変化
    plt.figure(figsize=(15, 8))
    motor_data = np.array([m['output'][0] for m in results['motor_outputs']])
    plt.imshow(motor_data.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='活性度')
    plt.title('運動出力の時間的変化')
    plt.ylabel('運動ニューロンID')
    plt.xlabel('時間ステップ')
    plt.savefig(os.path.join(output_dir, 'motor_outputs.png'))
    
    # 3. 予測と信頼度の時間的変化
    plt.figure(figsize=(15, 6))
    predictions = [p['prediction'] for p in results['predictions']]
    confidences = [p['confidence'] for p in results['predictions']]
    
    plt.subplot(2, 1, 1)
    plt.plot(range(len(predictions)), predictions, 'b-', marker='o')
    plt.title('予測クラスの時間的変化')
    plt.ylabel('予測クラス')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(range(len(confidences)), confidences, 'r-')
    plt.title('予測信頼度の時間的変化')
    plt.ylabel('信頼度')
    plt.xlabel('時間ステップ')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.png'))
    
    # 4. グリア細胞活動と結合強度
    plt.figure(figsize=(15, 10))
    for layer_idx in range(len(results['layer_activations'][0]['activations'])):
        plt.subplot(len(results['layer_activations'][0]['activations']), 1, layer_idx+1)
        
        astro_values = [a['activations'][layer_idx]['astrocyte_activity'] 
                        for a in results['layer_activations']]
        micro_values = [a['activations'][layer_idx]['microglia_activity'] 
                         for a in results['layer_activations']]
        conn_mean = [a['activations'][layer_idx]['connection_mean'] 
                     for a in results['layer_activations']]
        
        plt.plot(range(len(astro_values)), astro_values, label='アストロサイト活性')
        plt.plot(range(len(micro_values)), micro_values, label='ミクログリア活性')
        plt.plot(range(len(conn_mean)), conn_mean, label='平均結合強度')
        
        plt.title(f'レイヤー {layer_idx+1} のグリア細胞活動')
        plt.ylabel('活性度')
        plt.xlabel('時間ステップ')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'glial_activity.png'))
    
    # 結果をJSONに保存（後で詳細分析用）
    with open(os.path.join(output_dir, 'simulation_results.json'), 'w') as f:
        # NumPy配列をリストに変換するための関数
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_numpy(results), f, indent=2)
    
    print(f"解析結果を {output_dir} に保存しました")

# メイン実行
def main():
    print(f"非同期KANモデルシミュレーション開始 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"設定: {args}")
    
    # シミュレーション実行
    results = run_simulation(model, time_steps=args.time_steps)
    
    # 反実仮想シミュレーション
    x, _ = generate_input(batch_size=1, seq_len=100, dim=args.dim)
    counterfactuals = run_counterfactual_simulations(model, x)
    
    # 結果を保存
    with open(os.path.join(args.output_dir, 'counterfactuals.json'), 'w') as f:
        json.dump(counterfactuals, f, indent=2, default=lambda o: o.tolist() 
                  if isinstance(o, (np.ndarray, torch.Tensor)) else o)
    
    # 結果の可視化
    visualize_simulation_results(results, args.output_dir)
    
    print(f"シミュレーション完了 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"すべての結果は {args.output_dir} ディレクトリに保存されました")

if __name__ == "__main__":
    main() 