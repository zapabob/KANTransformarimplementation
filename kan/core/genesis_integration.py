import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import sys
import genesis
import time
# Genesisライブラリをインポートしようとするが、存在しない場合はモック実装を使用
try:
    import genesis
    import genesis.motor as gmotor
    GENESIS_AVAILABLE = True
except ImportError:
    GENESIS_AVAILABLE = False
    # IDE警告を避けるためにパスを明示的に作成
    import os
    import sys
    # ローカルのスタブディレクトリを確認
    stub_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'genesis')
    if os.path.exists(stub_dir) and stub_dir not in sys.path:
        sys.path.append(os.path.dirname(stub_dir))
    # これでgenesisがスタブとして見つかる可能性が高い
    try:
        import genesis
        import genesis.motor as gmotor
    except ImportError:
        print("警告: Genesisライブラリが見つかりません。モック実装を使用します。")
        # モッククラスの定義
        class MockMotorController:
            """
            Genesisライブラリが存在しない場合のモックコントローラ
            """
            def __init__(self, num_motors=10, **kwargs):
                self.num_motors = num_motors
                self.state = "初期化済み"
                self.current_positions = np.zeros(num_motors)
                self.target_positions = np.zeros(num_motors)
                self.velocities = np.zeros(num_motors)
                self.forces = np.zeros(num_motors)
                self.patterns = {}
                self.acetylcholine_level = 0.5
                self.active_pattern = None
                self.pattern_step = 0
                self.last_update = time.time()
                
            def set_motor_command(self, command, acetylcholine_level=0.5, **kwargs):
                """モーターコマンドを設定"""
                self.acetylcholine_level = acetylcholine_level
                
                # 精度、力、速度の計算
                precision = acetylcholine_level
                strength = 0.2 + (acetylcholine_level * 0.8)
                response_speed = 0.3 + (acetylcholine_level * 0.7)
                
                # コマンドを配列に変換
                if not isinstance(command, np.ndarray):
                    command = np.array(command)
                
                # サイズ調整
                if len(command) != self.num_motors:
                    command = np.resize(command, self.num_motors)
                
                # ノイズ適用
                noise_level = 0.3 * (1.0 - precision)
                noise = np.random.normal(0, noise_level, size=command.shape) if noise_level > 0.01 else 0
                
                # 最終コマンド
                self.target_positions = np.clip(command * strength + noise, -1.0, 1.0)
                
                # 現在の時間
                current_time = time.time()
                dt = current_time - self.last_update
                self.last_update = current_time
                
                # 位置の更新
                for i in range(self.num_motors):
                    diff = self.target_positions[i] - self.current_positions[i]
                    self.current_positions[i] += diff * response_speed * dt * 5.0
                    self.velocities[i] = diff * response_speed * 5.0
                
                print(f"モーターコマンド（モック）: 精度={precision:.2f}, 力={strength:.2f}, 速度={response_speed:.2f}")
                return True
                
            def register_motion_pattern(self, name, pattern):
                """運動パターンを登録"""
                self.patterns[name] = np.array(pattern)
                print(f"パターン登録（モック）: {name}, {len(pattern)}ステップ")
                return True
                
            def execute_motion(self, name, acetylcholine_level=0.5, **kwargs):
                """パターンを実行"""
                if name not in self.patterns:
                    print(f"エラー（モック）: パターン '{name}' は登録されていません")
                    return False
                
                self.acetylcholine_level = acetylcholine_level
                self.active_pattern = name
                self.pattern_step = 0
                
                # 最初のステップを実行
                pattern = self.patterns[name]
                if len(pattern) > 0:
                    self.set_motor_command(pattern[0], acetylcholine_level)
                
                print(f"パターン実行（モック）: {name}, アセチルコリン={acetylcholine_level:.2f}")
                return True
            
            def update(self, dt=0.01):
                """状態を更新"""
                if not self.active_pattern:
                    return
                
                pattern = self.patterns[self.active_pattern]
                
                # 進行速度
                speed = 0.5 + (self.acetylcholine_level * 0.5)
                self.pattern_step += dt * speed
                
                next_step = int(self.pattern_step)
                if next_step >= len(pattern):
                    self.active_pattern = None
                    return
                
                self.set_motor_command(pattern[next_step], self.acetylcholine_level)
                
            def get_state(self):
                """状態を取得"""
                return {
                    "status": "実行中（モック）",
                    "position": self.current_positions.copy(),
                    "target": self.target_positions.copy(),
                    "velocity": self.velocities.copy(),
                    "force": np.zeros_like(self.current_positions),
                    "acetylcholine": self.acetylcholine_level,
                    "active_pattern": self.active_pattern,
                    "pattern_step": self.pattern_step if self.active_pattern else 0
                }
        
        # モックモジュールの定義
        class MockGenesis:
            def __init__(self):
                self.motor = MockMotorController()
        
        # モックの設定
        genesis = MockGenesis()
        gmotor = genesis.motor


class GenesisMotorController:
    """
    Genesisライブラリを使用した運動制御モジュール
    - アセチルコリンレベルに応じた運動コマンドの生成
    - 事前に定義された運動パターンの実行
    - 運動状態のフィードバック
    """
    def __init__(self, num_motors: int = 10, pattern_library_path: Optional[str] = None):
        self.num_motors = num_motors
        self.pattern_library_path = pattern_library_path
        
        # モーターコントローラの初期化
        if GENESIS_AVAILABLE:
            self.motor_controller = gmotor.Controller(num_motors=num_motors)
        else:
            self.motor_controller = MockMotorController(num_motors=num_motors)
        
        # 運動パターンライブラリ
        self.patterns = {}
        
        # 現在の運動状態
        self.current_position = np.zeros(num_motors)
        self.target_position = np.zeros(num_motors)
        self.current_velocity = np.zeros(num_motors)
        
        # 運動パターンが実行中かどうか
        self.pattern_executing = False
        self.current_pattern = None
        self.pattern_step = 0
        
        # もしパスが指定されていれば、パターンを読み込む
        if pattern_library_path and os.path.exists(pattern_library_path):
            self._load_pattern_library(pattern_library_path)
            
        print(f"GenesisMotorController初期化: モーター数={num_motors}")
    
    def _load_pattern_library(self, path: str):
        """パターンライブラリをファイルから読み込む"""
        try:
            import json
            with open(path, 'r') as f:
                pattern_data = json.load(f)
                
            for name, data in pattern_data.items():
                pattern = [np.array(step) for step in data['steps']]
                self.patterns[name] = pattern
                
            print(f"{len(self.patterns)}個の運動パターンを読み込みました")
        except Exception as e:
            print(f"パターンライブラリの読み込みに失敗: {e}")
    
    def load_patterns(self, patterns: Dict[str, List[np.ndarray]]):
        """
        運動パターンをプログラムから直接読み込む
        patterns: パターン名をキー、ステップのリストを値とする辞書
        """
        for name, steps in patterns.items():
            self.patterns[name] = steps
            # Genesisの新APIでパターンを登録
            self.motor_controller.register_motion_pattern(name, steps)
        
        print(f"{len(patterns)}個の運動パターンを登録しました")
        return True
    
    def execute_command(self, 
                        command: Union[List[float], np.ndarray], 
                        acetylcholine_level: float = 0.5,
                        **kwargs) -> bool:
        """
        単一のモーターコマンドを実行する
        
        パラメータ:
        - command: モーターごとの目標位置（-1.0から1.0の範囲）
        - acetylcholine_level: アセチルコリンのレベル（0.0から1.0）
          - 高い値: 精度向上、素早い反応、より強い動き
          - 低い値: 精度低下、ゆっくりとした反応、弱い動き
        
        戻り値:
        - 成功した場合はTrue、失敗した場合はFalse
        """
        # 運動パターンの実行をキャンセル
        self.pattern_executing = False
        self.current_pattern = None
        
        # コマンドのバリデーション
        if not isinstance(command, (list, np.ndarray)):
            print(f"エラー: コマンドはリストまたはndarrayである必要があります")
            return False
        
        # リストの場合はndarrayに変換
        if isinstance(command, list):
            command = np.array(command, dtype=float)
        
        # サイズが一致しない場合は調整
        if len(command) != self.num_motors:
            resized_command = np.zeros(self.num_motors)
            min_len = min(len(command), self.num_motors)
            resized_command[:min_len] = command[:min_len]
            command = resized_command
            print(f"警告: コマンドサイズを調整しました（{len(command)} → {self.num_motors}）")
        
        # クリッピング
        command = np.clip(command, -1.0, 1.0)
        
        # アセチルコリンレベルを制限
        acetylcholine_level = np.clip(acetylcholine_level, 0.0, 1.0)
        
        # 運動コマンドの実行
        try:
            success = self.motor_controller.set_motor_command(
                command, 
                acetylcholine_level=acetylcholine_level,
                **kwargs
            )
            
            # 状態の更新
            self.target_position = command.copy()
            
            return success
        except Exception as e:
            print(f"運動コマンド実行中にエラーが発生: {e}")
            return False
    
    def execute_pattern(self, 
                        pattern_name: str, 
                        acetylcholine_level: float = 0.5,
                        **kwargs) -> bool:
        """
        登録済みの運動パターンを実行する
        
        パラメータ:
        - pattern_name: 実行するパターンの名前
        - acetylcholine_level: アセチルコリンのレベル（0.0から1.0）
          - 高い値: 精度向上、素早い実行、より強い動き
          - 低い値: 精度低下、ゆっくりとした実行、弱い動き
        
        戻り値:
        - 成功した場合はTrue、失敗した場合はFalse
        """
        # パターンがライブラリに存在するか確認
        if pattern_name not in self.patterns:
            print(f"エラー: パターン '{pattern_name}' が見つかりません")
            return False
        
        # アセチルコリンレベルを制限
        acetylcholine_level = np.clip(acetylcholine_level, 0.0, 1.0)
        
        # 新API: 直接パターン実行メソッドを呼び出し
        try:
            success = self.motor_controller.execute_motion(
                pattern_name, 
                acetylcholine_level=acetylcholine_level,
                **kwargs
            )
            
            if success:
                self.pattern_executing = True
                self.current_pattern = pattern_name
                self.pattern_step = 0
                print(f"パターン実行開始: {pattern_name}, アセチルコリン={acetylcholine_level:.2f}")
            
            return success
        except Exception as e:
            print(f"パターン実行中にエラーが発生: {e}")
            return False
    
    def update(self, dt: float = 0.01) -> None:
        """
        モーターコントローラの状態を更新する
        
        パラメータ:
        - dt: 前回の更新からの経過時間（秒）
        """
        if hasattr(self.motor_controller, 'update'):
            self.motor_controller.update(dt)
            
        # 現在の状態を取得
        state = self.motor_controller.get_state()
        
        # 状態の更新
        if 'position' in state:
            self.current_position = state['position']
        if 'velocity' in state:
            self.current_velocity = state.get('velocity', np.zeros_like(self.current_position))
        
        # パターン実行状態の更新
        if 'active_pattern' in state:
            self.current_pattern = state.get('active_pattern')
            self.pattern_executing = self.current_pattern is not None
            self.pattern_step = state.get('pattern_step', 0)
    
    def get_state(self) -> Dict:
        """
        現在の運動状態を取得する
        
        戻り値:
        - 状態を含む辞書
          - position: 現在のモーター位置
          - target: 目標位置
          - velocity: 現在の速度
          - pattern_executing: パターンが実行中かどうか
          - current_pattern: 現在実行中のパターン名
          - pattern_step: 現在のパターンステップ
        """
        try:
            motor_state = self.motor_controller.get_state()
            
            # 追加情報
            motor_state.update({
                "pattern_executing": self.pattern_executing,
                "current_pattern": self.current_pattern,
                "pattern_step": self.pattern_step
            })
            
            return motor_state
        except Exception as e:
            print(f"状態取得中にエラーが発生: {e}")
            return {
                "error": str(e),
                "pattern_executing": self.pattern_executing,
                "current_pattern": self.current_pattern,
                "pattern_step": self.pattern_step
            }


class MotorCortexLayer(nn.Module):
    """
    運動皮質層: 神経活動から運動コマンドを生成
    - アセチルコリンによる運動出力の調整
    - 運動パターンのエンコーディングと認識
    """
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int = 10, 
                 hidden_dim: int = 128):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # ニューラルネットワーク層
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        # 運動パターンの埋め込み表現
        self.pattern_embeddings = {}
        
        # モーターコントローラ（実際の運動出力用）
        self.motor_controller = GenesisMotorController(num_motors=output_dim)
        
        # 内部状態
        self.last_output = torch.zeros(output_dim)
        self.output_history = []
        self.max_history = 100
    
    def forward(self, x: torch.Tensor, acetylcholine_level: float = 0.5) -> torch.Tensor:
        """
        x: 入力テンソル [batch_size, input_dim]
        acetylcholine_level: アセチルコリンレベル（0.0～1.0）
        returns: 運動出力テンソル [batch_size, output_dim]
        """
        # 入力の形状をチェック
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # バッチ次元を追加
        
        # 前方伝播
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout1(h1)
        h2 = F.relu(self.fc2(h1))
        h2 = self.dropout2(h2)
        
        # アセチルコリンレベルに基づく出力調整
        # - 高レベル: より精確な運動指令（tanh出力）
        # - 低レベル: よりぼんやりとした運動指令（sigmoid出力で正の値のみ）
        if acetylcholine_level > 0.7:
            # 高アセチルコリン: 明確な運動指令
            motor_output = torch.tanh(self.fc3(h2))
        elif acetylcholine_level > 0.3:
            # 中程度アセチルコリン: 混合出力
            tanh_output = torch.tanh(self.fc3(h2))
            sigmoid_output = torch.sigmoid(self.fc3(h2)) * 2 - 1
            motor_output = acetylcholine_level * tanh_output + (1 - acetylcholine_level) * sigmoid_output
        else:
            # 低アセチルコリン: ぼんやりとした運動指令
            motor_output = torch.sigmoid(self.fc3(h2)) * 2 - 1
        
        # 履歴の更新
        self.last_output = motor_output.detach().mean(0)
        self.output_history.append(self.last_output.cpu().numpy())
        if len(self.output_history) > self.max_history:
            self.output_history.pop(0)
        
        return motor_output
    
    def execute_motor_command(self, output: torch.Tensor, acetylcholine_level: float = 0.5):
        """
        モーターコントローラを使用して実際の運動コマンドを実行
        output: 運動出力テンソル
        acetylcholine_level: アセチルコリンレベル
        """
        # テンソルをNumPy配列に変換
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
            
        # バッチ次元がある場合は最初の要素を使用
        if len(output.shape) > 1:
            output = output[0]
        
        # モーターコマンドの実行
        self.motor_controller.execute_command(output, acetylcholine_level)
    
    def load_patterns(self, patterns: Dict[str, List[np.ndarray]]):
        """
        運動パターンをモーターコントローラに読み込み、埋め込み表現を計算
        patterns: パターン名をキー、ステップのリストを値とする辞書
        """
        # モーターコントローラにパターンを読み込む
        self.motor_controller.load_patterns(patterns)
        
        # 各パターンの埋め込み表現を計算
        for name, steps in patterns.items():
            # パターンの平均ベクトルを計算
            pattern_avg = np.mean(steps, axis=0)
            
            # パターンの埋め込み表現を作成（単純な平均ベクトル）
            self.pattern_embeddings[name] = torch.tensor(pattern_avg, dtype=torch.float32)
    
    def recognize_pattern(self, output: torch.Tensor) -> Tuple[str, float]:
        """
        出力された運動コマンドから最も近いパターンを認識
        output: 運動出力テンソル
        returns: (最も近いパターン名, 類似度スコア)
        """
        if len(self.pattern_embeddings) == 0:
            return "不明", 0.0
        
        # 出力をNumPy配列に変換
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
            
        # バッチ次元がある場合は最初の要素を使用
        if len(output.shape) > 1:
            output = output[0]
        
        output_tensor = torch.tensor(output, dtype=torch.float32)
        
        # 各パターンとの類似度を計算
        similarities = {}
        for name, embedding in self.pattern_embeddings.items():
            # コサイン類似度を計算
            similarity = F.cosine_similarity(
                output_tensor.unsqueeze(0), 
                embedding.unsqueeze(0)
            ).item()
            similarities[name] = similarity
        
        # 最も類似度の高いパターンを返す
        best_pattern = max(similarities.items(), key=lambda x: x[1])
        return best_pattern[0], best_pattern[1]
    
    def get_state(self) -> Dict[str, Any]:
        """モジュールの現在の状態を取得"""
        motor_state = self.motor_controller.get_state()
        
        # 最近の出力履歴から活動統計を計算
        if len(self.output_history) > 0:
            output_history_np = np.array(self.output_history)
            activity_stats = {
                'mean': np.mean(output_history_np),
                'std': np.std(output_history_np),
                'min': np.min(output_history_np),
                'max': np.max(output_history_np)
            }
        else:
            activity_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        return {
            'last_output': self.last_output.cpu().numpy().tolist(),
            'activity_stats': activity_stats,
            'motor_controller': motor_state
        }


class CorticalLayerStructure(nn.Module):
    """
    皮質層構造: 階層的な情報処理を模倣
    - 大脳皮質の6層構造を模倣
    - 層ごとの異なる情報処理特性
    """
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 num_layers: int = 6,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # 皮質層（6層構造）
        self.layers = nn.ModuleList([
            # 層I: 表層神経細胞
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            # 層II: 粒上層
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            # 層III: 外錐体層
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            # 層IV: 内顆粒層
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            # 層V: 内錐体層
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            # 層VI: 多形層
            nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.Tanh()
            )
        ])
        
        # 層間スキップ接続
        self.skip_connections = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),  # 入力 -> 層II
            nn.Linear(input_dim, hidden_dim),  # 入力 -> 層III
            nn.Linear(hidden_dim, hidden_dim), # 層II -> 層IV
            nn.Linear(hidden_dim, hidden_dim), # 層III -> 層V
            nn.Linear(hidden_dim, output_dim)  # 層IV -> 層VI
        ])
    
    def forward(self, x: torch.Tensor, neuromod_state: Dict[str, float]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        x: 入力テンソル [batch_size, sequence_length, input_dim]
        neuromod_state: 神経調節物質の状態
        returns: (出力テンソル, 各層の出力リスト)
        """
        layer_outputs = []
        
        # 皮質層を通して前方伝播
        h = x
        
        # 神経調節状態による各層の挙動調整係数
        layer_factors = [
            1.0,  # 層I: 標準
            1.0 + 0.2 * neuromod_state['acetylcholine'],  # 層II: アセチルコリンで活性化
            1.0 + 0.3 * neuromod_state['glutamate'],      # 層III: グルタミン酸で活性化
            1.0 - 0.2 * neuromod_state['gaba'],           # 層IV: GABAで抑制
            1.0 + 0.3 * neuromod_state['dopamine'],       # 層V: ドーパミンで活性化
            1.0                                           # 層VI: 標準
        ]
        
        # スキップ接続の値を計算
        skip1 = self.skip_connections[0](x)  # 入力 -> 層II
        skip2 = self.skip_connections[1](x)  # 入力 -> 層III
        
        # 層I
        h = self.layers[0](h) * layer_factors[0]
        layer_outputs.append(h)
        
        # 層II（スキップ接続あり）
        h = self.layers[1](h + skip1) * layer_factors[1]
        layer_outputs.append(h)
        
        # 層III（スキップ接続あり）
        h = self.layers[2](h + skip2) * layer_factors[2]
        layer_outputs.append(h)
        skip3 = self.skip_connections[2](h)  # 層II -> 層IV
        
        # 層IV（スキップ接続あり）
        h = self.layers[3](h + skip3) * layer_factors[3]
        layer_outputs.append(h)
        skip4 = self.skip_connections[3](h)  # 層III -> 層V
        
        # 層V（スキップ接続あり）
        h = self.layers[4](h + skip4) * layer_factors[4]
        layer_outputs.append(h)
        skip5 = self.skip_connections[4](h)  # 層IV -> 層VI
        
        # 層VI（スキップ接続あり）
        h = self.layers[5](h + skip5) * layer_factors[5]
        layer_outputs.append(h)
        
        return h, layer_outputs 