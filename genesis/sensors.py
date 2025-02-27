"""
Genesis センサーモジュール

このモジュールは様々な感覚モダリティに対応するセンサーをシミュレーションします。
視覚、触覚、聴覚、内受容感覚などの感覚入力をエージェントに提供します。

主な機能:
- 視覚センサー（画像入力）
- 触覚センサー（圧力、接触）
- 固有受容感覚（姿勢、関節角度）
- 前庭感覚（平衡、加速度）

詳細なドキュメント: https://genesis-world.readthedocs.io/en/latest/modules/sensors.html
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union, Any

class SensorBase:
    """
    すべてのセンサーの基底クラス
    """
    def __init__(self, name: str, dimensions: Tuple[int, ...], update_rate: float = 30.0):
        """
        センサーを初期化
        
        パラメータ:
        - name: センサー名
        - dimensions: センサー出力の次元（タプル）
        - update_rate: 更新レート（Hz）
        """
        self.name = name
        self.dimensions = dimensions
        self.update_rate = update_rate
        self.data = np.zeros(dimensions)
        self.last_update = time.time()
        
        print(f"センサー初期化: {name}, 次元={dimensions}, 更新レート={update_rate}Hz")
    
    def update(self) -> np.ndarray:
        """
        センサーデータを更新して返す
        
        戻り値:
        - 更新されたセンサーデータ
        """
        current_time = time.time()
        dt = current_time - self.last_update
        
        # 更新間隔が短すぎる場合はスキップ
        if dt < 1.0 / self.update_rate:
            return self.data
        
        self.last_update = current_time
        
        # オーバーライドされるメソッド
        return self.data
    
    def get_data(self) -> np.ndarray:
        """
        現在のセンサーデータを取得
        
        戻り値:
        - センサーデータ
        """
        return self.data.copy()


class VisionSensor(SensorBase):
    """
    視覚センサー - 画像データを提供
    """
    def __init__(self, resolution: Tuple[int, int] = (64, 64), channels: int = 3, 
                 fov: float = 90.0, update_rate: float = 30.0):
        """
        視覚センサーを初期化
        
        パラメータ:
        - resolution: 画像解像度 (幅, 高さ)
        - channels: 色チャンネル数（1=グレースケール、3=RGB）
        - fov: 視野角（度）
        - update_rate: 更新レート（Hz）
        """
        super().__init__(
            name="視覚",
            dimensions=(resolution[1], resolution[0], channels),
            update_rate=update_rate
        )
        self.resolution = resolution
        self.channels = channels
        self.fov = fov
        
        # 視界障害の設定
        self.noise_level = 0.05
        self.blur_level = 0.0
        
    def update(self) -> np.ndarray:
        """視覚データを更新して返す"""
        super().update()
        
        # モック実装ではノイズのある画像を生成
        self.data = np.random.random(self.dimensions) * 0.1
        
        # 中央に十字パターン（簡易フィクセーション）
        center_x = self.resolution[0] // 2
        center_y = self.resolution[1] // 2
        width = max(2, self.resolution[0] // 16)
        
        # 水平線
        self.data[center_y-width//2:center_y+width//2, :, :] = 0.8
        # 垂直線
        self.data[:, center_x-width//2:center_x+width//2, :] = 0.8
        
        # ノイズを追加
        self.data += np.random.normal(0, self.noise_level, self.dimensions)
        self.data = np.clip(self.data, 0.0, 1.0)
        
        return self.data
    
    def set_blur(self, level: float) -> None:
        """
        視覚のぼやけレベルを設定
        
        パラメータ:
        - level: ぼやけのレベル（0.0～1.0）
        """
        self.blur_level = np.clip(level, 0.0, 1.0)


class TactileSensor(SensorBase):
    """
    触覚センサー - 圧力や触覚刺激を検出
    """
    def __init__(self, num_receptors: int = 100, sensitivity: float = 1.0, 
                 adaptation_rate: float = 0.5, update_rate: float = 60.0):
        """
        触覚センサーを初期化
        
        パラメータ:
        - num_receptors: 受容器の数
        - sensitivity: 感度（0.0～1.0）
        - adaptation_rate: 順応率（0.0～1.0）
        - update_rate: 更新レート（Hz）
        """
        super().__init__(
            name="触覚",
            dimensions=(num_receptors,),
            update_rate=update_rate
        )
        self.num_receptors = num_receptors
        self.sensitivity = sensitivity
        self.adaptation_rate = adaptation_rate
        
        # 前回の値（順応モデル用）
        self.previous_data = np.zeros(num_receptors)
    
    def update(self) -> np.ndarray:
        """触覚データを更新して返す"""
        super().update()
        
        # モック実装ではランダムな触覚刺激を生成
        raw_input = np.random.random(self.num_receptors) * 0.2
        
        # いくつかのランダムな受容器に強い刺激を与える
        active_receptors = np.random.choice(
            self.num_receptors, 
            size=max(1, self.num_receptors // 10), 
            replace=False
        )
        raw_input[active_receptors] = np.random.random(len(active_receptors)) * 0.8 + 0.2
        
        # 感度による調整
        scaled_input = raw_input * self.sensitivity
        
        # 順応モデル（持続的な刺激に対する反応の減少）
        delta = scaled_input - self.previous_data
        adaptation = self.previous_data * self.adaptation_rate
        
        # 新しい値の計算
        self.data = self.previous_data + delta - adaptation
        self.data = np.clip(self.data, 0.0, 1.0)
        
        # 現在値を記録
        self.previous_data = self.data.copy()
        
        return self.data


class ProprioceptionSensor(SensorBase):
    """
    固有受容感覚 - 関節角度や筋肉の状態を検出
    """
    def __init__(self, num_joints: int = 10, update_rate: float = 100.0):
        """
        固有受容感覚センサーを初期化
        
        パラメータ:
        - num_joints: 関節数
        - update_rate: 更新レート（Hz）
        """
        super().__init__(
            name="固有受容感覚",
            dimensions=(num_joints, 2),  # 位置と速度
            update_rate=update_rate
        )
        self.num_joints = num_joints
        
        # 関節の制限
        self.joint_limits = np.array([(-1.0, 1.0) for _ in range(num_joints)])
    
    def update(self) -> np.ndarray:
        """固有受容感覚データを更新して返す"""
        super().update()
        
        # モック実装では現実的な関節運動をシミュレート
        dt = time.time() - self.last_update
        
        # 位置の更新: 現在の位置からランダムな変化（速度に比例）
        position_delta = self.data[:, 1] * dt + np.random.normal(0, 0.01, self.num_joints)
        new_positions = self.data[:, 0] + position_delta
        
        # 関節制限の適用
        for i in range(self.num_joints):
            lower, upper = self.joint_limits[i]
            if new_positions[i] < lower:
                new_positions[i] = lower
            elif new_positions[i] > upper:
                new_positions[i] = upper
        
        # 速度の更新: ランダムな加速度を適用
        new_velocities = self.data[:, 1] + np.random.normal(0, 0.02, self.num_joints)
        new_velocities *= 0.95  # 減衰
        
        # データの更新
        self.data[:, 0] = new_positions
        self.data[:, 1] = new_velocities
        
        return self.data


class VestibularSensor(SensorBase):
    """
    前庭感覚 - 平衡感覚、加速度、回転を検出
    """
    def __init__(self, update_rate: float = 100.0):
        """
        前庭感覚センサーを初期化
        
        パラメータ:
        - update_rate: 更新レート（Hz）
        """
        super().__init__(
            name="前庭感覚",
            dimensions=(6,),  # 3軸線形加速度, 3軸角速度
            update_rate=update_rate
        )
        
        # 最大値
        self.max_linear_accel = 9.8 * 3  # m/s^2, 重力の3倍
        self.max_angular_vel = np.pi * 2  # rad/s, 360度/秒
        
        # 重力
        self.gravity = np.array([0, 0, -9.8])
    
    def update(self) -> np.ndarray:
        """前庭感覚データを更新して返す"""
        super().update()
        
        # モック実装ではランダムな動きをシミュレート
        
        # 線形加速度（重力を含む）
        linear_accel = np.random.normal(0, 0.2, 3)
        linear_accel[2] -= 9.8  # Z軸の重力
        
        # 角速度
        angular_vel = np.random.normal(0, 0.1, 3)
        
        # データの更新
        self.data[0:3] = linear_accel  # 線形加速度
        self.data[3:6] = angular_vel   # 角速度
        
        return self.data


def create_sensor(sensor_type: str, **kwargs) -> SensorBase:
    """
    指定された種類のセンサーを作成
    
    パラメータ:
    - sensor_type: センサーの種類（'vision', 'tactile', 'proprioception', 'vestibular'）
    - **kwargs: センサー固有のパラメータ
    
    戻り値:
    - 作成されたセンサーオブジェクト
    """
    sensor_map = {
        'vision': VisionSensor,
        'tactile': TactileSensor,
        'proprioception': ProprioceptionSensor,
        'vestibular': VestibularSensor
    }
    
    if sensor_type not in sensor_map:
        raise ValueError(f"未知のセンサータイプ: {sensor_type}")
    
    return sensor_map[sensor_type](**kwargs) 