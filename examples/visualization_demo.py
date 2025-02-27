from kan.core import KANTransformer
from kan.utils import KANVisualizer, create_dashboard

def main():
    # モデルのパスを指定（トレーニング後のチェックポイント）
    model_path = 'checkpoints/latest.pt'
    
    # 可視化クラスの初期化
    visualizer = KANVisualizer(model_path)
    
    # ダッシュボードの作成と実行
    app = create_dashboard(visualizer)
    app.run_server(debug=True)

if __name__ == '__main__':
    main() 