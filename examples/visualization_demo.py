import torch
import matplotlib.pyplot as plt
import seaborn as sns
from kan.core import KANTransformer
from kan.data import create_dataloaders

def plot_neuromod_states(model, save_path='neuromod_states.png'):
    """神経調節状態の可視化"""
    states = model.get_neuromod_visualization()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = range(len(states['dopamine']))
    
    plt.plot(layers, states['dopamine'], 'r-', label='ドーパミン')
    plt.plot(layers, states['noradrenaline'], 'b-', label='ノルアドレナリン')
    plt.plot(layers, states['serotonin'], 'g-', label='セロトニン')
    
    plt.xlabel('レイヤー')
    plt.ylabel('活性化レベル')
    plt.title('神経調節状態の推移')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_attention_patterns(model, sample_input, save_path='attention_patterns.png'):
    """注意機構パターンの可視化"""
    model.eval()
    with torch.no_grad():
        _ = model(sample_input)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, attn_map in enumerate(model.attention_maps[:6]):  # 最初の6レイヤーのみ表示
        ax = axes[i//3, i%3]
        sns.heatmap(attn_map[0].cpu(), ax=ax, cmap='viridis')
        ax.set_title(f'Layer {i+1}')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # モデルとデータの準備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KANTransformer().to(device)
    train_loader, _ = create_dataloaders(batch_size=4)
    
    # サンプルデータの取得
    sample_batch = next(iter(train_loader))
    inputs, _ = sample_batch
    inputs = inputs.to(device)
    
    # 推論と可視化
    explanation = model.get_explanation(inputs)
    print(f"予測クラス: {explanation['prediction']}")
    print(f"信頼度: {explanation['confidence']:.4f}")
    
    # 可視化の実行
    plot_neuromod_states(model)
    plot_attention_patterns(model, inputs)

if __name__ == '__main__':
    main() 