"""
Genesis モーターコントロールモジュール

このモジュールは物理エージェントの運動制御機能を提供します。
神経修飾物質（特にアセチルコリン）の効果を考慮した運動制御をシミュレーションします。

主な機能:
- モーターコマンドの生成と実行
- 神経修飾物質の影響を受ける運動パラメータの調整
- 運動パターンの登録と実行
- モーター状態のモニタリング

詳細なドキュメント: https://genesis-world.readthedocs.io/en/latest/modules/motor.html
"""

import numpy as np
import time
import random
import threading
from typing import Dict, List, Tuple, Optional, Union, Any

# Genesisがインストールされているか確認し、適切にインポート
try:
    import genesis as gs
    from genesis.core import Motor, Joint
    from genesis.utils import interpolate
    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False
    # モックオブジェクトを使用
    print("注意: genesis-worldがインストールされていません。モック実装を使用します。")

# アセチルコリンレベルによる運動特性の影響係数
ACH_PRECISION_FACTOR = 0.7  # 精度への影響度
ACH_STRENGTH_FACTOR = 0.8   # 力への影響度
ACH_SPEED_FACTOR = 0.6      # 速度への影響度
ACH_SMOOTHNESS_FACTOR = 0.5 # 滑らかさへの影響度

# デフォルト設定
DEFAULT_CONFIG = {
    "motor_update_rate": 60.0,  # Hz
    "motor_noise_level": 0.05,  # 標準偏差
    "acetylcholine_default": 0.5,  # デフォルトレベル
    "response_curve": "sigmoid",  # シグモイド、線形、または指数関数
    "simulated_delay": 0.01,  # 秒
}

class Controller:
    """
    モーターコントローラークラス
    
    このクラスは筋肉の収縮と運動出力を制御し、アセチルコリンレベルに応じた
    さまざまな運動特性の調整を行います。
    
    アセチルコリンの影響:
    - 精度: 高アセチルコリン = 高精度（運動ノイズ減少）
    - 力: 高アセチルコリン = 強い筋肉収縮（振幅増大）
    - 応答速度: 高アセチルコリン = 素早い筋肉応答（遅延減少）
    - 滑らかさ: 高アセチルコリン = 滑らかな運動（振動減少）
    """
    
    def __init__(self, num_motors: int = 10, config: Optional[Dict[str, Any]] = None):
        """
        モーターコントローラーを初期化
        
        パラメータ:
        - num_motors: モーターユニットの数
        - config: 設定パラメータ
        """
        self.num_motors = num_motors
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # モーター状態
        self.current_positions = np.zeros(num_motors)
        self.target_positions = np.zeros(num_motors)
        self.velocities = np.zeros(num_motors)
        self.forces = np.zeros(num_motors)
        
        # アセチルコリンレベル
        self.acetylcholine_level = self.config["acetylcholine_default"]
        
        # モーションパターンライブラリ
        self.patterns = {}
        
        # 実行状態
        self.active_pattern = None
        self.pattern_step = 0
        self.last_update_time = time.time()
        
        # モーター特性の計算用係数
        self._recalculate_motor_coefficients()
        
        # 状態監視用のスレッド（ここでは単純化）
        self._running = True
        self._update_thread = None
        self._start_update_thread()
        
        print(f"モーターコントローラー初期化: ユニット数={num_motors}")
        
        # Genesis APIが存在する場合はモーターを初期化
        if HAS_GENESIS:
            self.motors = [Motor(id=i, max_force=self.config["max_force"], damping=self.config["damping"]) for i in range(num_motors)]
            self.env = gs.Environment()
            for motor in self.motors:
                self.env.add_object(motor)
        else:
            # モックモーター
            self.motors = [{"id": i, "pos": 0.0, "vel": 0.0} for i in range(num_motors)]
            self.last_update = time.time()
    
    def _recalculate_motor_coefficients(self):
        """アセチルコリンレベルに基づいて運動特性の係数を計算"""
        ach = self.acetylcholine_level
        
        # 各特性の基本計算（0.0～1.0の範囲に正規化）
        
        # 1. 精度: 高ACh = 低ノイズ
        self.precision = 0.3 + (0.7 * ach)  # 0.3～1.0
        self.noise_factor = 1.0 - self.precision
        
        # 2. 力: 高ACh = 強い力
        self.strength = 0.2 + (0.8 * ach)  # 0.2～1.0
        
        # 3. 応答速度: 高ACh = 素早い応答
        self.response_speed = 0.3 + (0.7 * ach)  # 0.3～1.0
        
        # 4. 滑らかさ: 高ACh = 滑らかな動き
        self.smoothness = 0.4 + (0.6 * ach)  # 0.4～1.0
        self.jitter_factor = 1.0 - self.smoothness
    
    def _start_update_thread(self):
        """モーター状態更新用のバックグラウンドスレッドを開始"""
        if self._update_thread is not None and self._update_thread.is_alive():
            return
        
        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop)
        self._update_thread.daemon = True
        self._update_thread.start()
    
    def _update_loop(self):
        """モーター状態更新ループ（バックグラウンドスレッド）"""
        update_interval = 1.0 / self.config["motor_update_rate"]
        
        while self._running:
            self.update(update_interval)
            time.sleep(update_interval)
    
    def set_motor_command(self, command: Union[List[float], np.ndarray], 
                          acetylcholine_level: Optional[float] = None, **kwargs) -> bool:
        """
        モーターコマンドを設定
        
        パラメータ:
        - command: モーターコマンド配列（-1.0～1.0）、長さはnum_motors
        - acetylcholine_level: アセチルコリンレベル（0.0～1.0）
        - **kwargs: 追加パラメータ
        
        戻り値:
        - 成功した場合はTrue
        """
        # アセチルコリンレベルの更新
        if acetylcholine_level is not None:
            self.acetylcholine_level = np.clip(acetylcholine_level, 0.0, 1.0)
            self._recalculate_motor_coefficients()
        
        # コマンドの変換
        if not isinstance(command, np.ndarray):
            command = np.array(command, dtype=float)
        
        # サイズ調整
        if len(command) != self.num_motors:
            resized_command = np.zeros(self.num_motors)
            min_len = min(len(command), self.num_motors)
            resized_command[:min_len] = command[:min_len]
            command = resized_command
        
        # アセチルコリンの影響を適用
        
        # 1. 精度: ノイズレベルを計算
        base_noise = self.config["motor_noise_level"]
        effective_noise = base_noise * self.noise_factor
        
        if effective_noise > 0.001:  # 無視できないノイズレベルの場合
            noise = np.random.normal(0, effective_noise, size=command.shape)
        else:
            noise = 0
        
        # 2. 力: コマンド振幅を調整
        strength_adjusted = command * self.strength
        
        # 3 & 4. 応答速度と滑らかさ: 現在の更新では直接使わない
        # アクチュエータのシミュレーションで使用される
        
        # 最終コマンドの計算（ノイズと強度を適用）
        final_command = np.clip(strength_adjusted + noise, -1.0, 1.0)
        
        # 目標位置を更新
        self.target_positions = final_command.copy()
        
        # 実行中のパターンをキャンセル
        self.active_pattern = None
        
        # デバッグ情報
        debug_info = {
            "ach_level": self.acetylcholine_level,
            "precision": self.precision,
            "strength": self.strength,
            "response": self.response_speed,
            "smoothness": self.smoothness
        }
        
        # コマンド設定ログ
        print(f"モーターコマンド設定: ACh={self.acetylcholine_level:.2f}, "
              f"精度={self.precision:.2f}, 力={self.strength:.2f}")
        
        return True
    
    def register_motion_pattern(self, name: str, pattern: Union[List[List[float]], List[np.ndarray], np.ndarray]) -> bool:
        """
        運動パターンをライブラリに登録
        
        パラメータ:
        - name: パターン名
        - pattern: モーターコマンドのシーケンス
        
        戻り値:
        - 成功した場合はTrue
        """
        # パターンの検証
        if not pattern or len(pattern) == 0:
            print(f"エラー: パターン '{name}' は空です")
            return False
        
        # パターンを配列のリストとして保存
        processed_pattern = []
        for step in pattern:
            if isinstance(step, list):
                step = np.array(step, dtype=float)
            
            # サイズ調整
            if len(step) != self.num_motors:
                resized_step = np.zeros(self.num_motors)
                min_len = min(len(step), self.num_motors)
                resized_step[:min_len] = step[:min_len]
                step = resized_step
            
            processed_pattern.append(step)
        
        # パターンを保存
        self.patterns[name] = processed_pattern
        print(f"パターン登録: '{name}', {len(processed_pattern)}ステップ")
        
        return True
    
    def execute_motion(self, name: str, acetylcholine_level: Optional[float] = None, 
                       loop: bool = False, **kwargs) -> bool:
        """
        登録済みの運動パターンを実行
        
        パラメータ:
        - name: 実行するパターンの名前
        - acetylcholine_level: アセチルコリンレベル（0.0～1.0）
        - loop: パターンをループするかどうか
        - **kwargs: 追加パラメータ
        
        戻り値:
        - 成功した場合はTrue
        """
        # パターンの存在確認
        if name not in self.patterns:
            print(f"エラー: パターン '{name}' が見つかりません")
            return False
        
        # アセチルコリンレベルの更新
        if acetylcholine_level is not None:
            self.acetylcholine_level = np.clip(acetylcholine_level, 0.0, 1.0)
            self._recalculate_motor_coefficients()
        
        # パターン実行状態を設定
        self.active_pattern = name
        self.pattern_step = 0
        self.is_looping = loop
        
        # 最初のステップを実行
        pattern = self.patterns[name]
        if len(pattern) > 0:
            self.set_motor_command(pattern[0], self.acetylcholine_level)
        
        print(f"パターン実行開始: '{name}', ACh={self.acetylcholine_level:.2f}, ループ={loop}")
        return True
    
    def update(self, dt: float = 0.01) -> None:
        """
        モーター状態を更新
        
        パラメータ:
        - dt: 前回の更新からの経過時間（秒）
        """
        # 現在のターゲットと位置の差を計算
        position_diff = self.target_positions - self.current_positions
        
        # 応答速度に基づいて位置を更新
        # アセチルコリンレベルが高いほど速く目標位置に近づく
        update_rate = dt * 10.0 * self.response_speed
        update_step = position_diff * update_rate
        
        # 滑らかさに基づくジッターの追加
        if self.jitter_factor > 0.01:
            jitter = np.random.normal(0, 0.02 * self.jitter_factor, size=position_diff.shape)
            update_step += jitter
        
        # 位置の更新
        self.current_positions += update_step
        self.current_positions = np.clip(self.current_positions, -1.0, 1.0)
        
        # 速度の更新（単純な近似）
        self.velocities = update_step / dt
        
        # アクティブなパターンの進行
        if self.active_pattern is not None:
            pattern = self.patterns[self.active_pattern]
            
            # パターン進行速度（アセチルコリンに影響される）
            step_increment = dt * (1.0 + self.response_speed)
            self.pattern_step += step_increment
            
            # 次のステップに進むべきかを確認
            next_step_idx = int(self.pattern_step)
            
            # パターンの終了またはループ
            if next_step_idx >= len(pattern):
                if self.is_looping:
                    # ループ - パターンを再開
                    self.pattern_step = 0
                    next_step_idx = 0
                else:
                    # 終了
                    self.active_pattern = None
                    return
            
            # 次のステップを設定（アセチルコリンレベルは既に設定済み）
            self.set_motor_command(pattern[next_step_idx])
    
    def get_state(self) -> Dict[str, Any]:
        """
        現在のモーター状態を取得
        
        戻り値:
        - 状態情報を含む辞書
        """
        # Genesis APIが存在する場合は実際のモーター状態を取得
        if HAS_GENESIS:
            for i, motor in enumerate(self.motors):
                self.current_positions[i] = motor.get_position()
                self.velocities[i] = motor.get_velocity()
                self.forces[i] = motor.get_force()
        
        return {
            "status": "実行中",
            "position": self.current_positions.copy(),
            "target": self.target_positions.copy(),
            "velocity": self.velocities.copy(),
            "force": self.forces.copy(),
            "acetylcholine": self.acetylcholine_level,
            "active_pattern": self.active_pattern,
            "pattern_step": self.pattern_step if self.active_pattern else 0,
            "motor_parameters": {
                "precision": self.precision,
                "strength": self.strength,
                "response_speed": self.response_speed,
                "smoothness": self.smoothness
            }
        }
    
    def stop(self) -> bool:
        """
        すべてのモーター活動を停止
        
        戻り値:
        - 成功した場合はTrue
        """
        self.active_pattern = None
        self.target_positions = np.zeros(self.num_motors)
        self._running = False
        
        # 更新スレッドが存在する場合は終了を待機
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)
        
        print("モーターコントローラー停止")
        return True

# Genesis MotorControllerのエイリアス（互換性のため）
MotorController = Controller 