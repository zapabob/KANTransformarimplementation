"""
Genesis 環境モジュール

このモジュールは物理ベースのシミュレーション環境を提供します。
エージェントが操作可能なオブジェクトや、様々な物理特性を持つ世界を構築できます。

主な機能:
- 物理オブジェクトの作成と操作
- 環境の物理シミュレーション
- 重力、摩擦、衝突などの物理法則の適用
- オブジェクト間の相互作用

詳細なドキュメント: https://genesis-world.readthedocs.io/en/latest/modules/environment.html
"""

import numpy as np
import time
import random
from typing import Dict, List, Tuple, Optional, Union, Any

# デフォルト設定
DEFAULT_CONFIG = {
    "gravity": [0, 0, -9.81],  # m/s^2
    "timestep": 0.01,          # 10ms
    "solver_iterations": 10,   # 物理ソルバーの反復回数
    "max_objects": 1000,       # 最大オブジェクト数
}

class PhysicalObject:
    """
    物理環境内のオブジェクトを表すクラス
    
    物理特性（質量、摩擦、弾性など）と状態（位置、回転、速度）を持ちます。
    """
    
    def __init__(self, object_id: int, name: str = "Object", 
                 mass: float = 1.0, position: List[float] = None, 
                 rotation: List[float] = None):
        """
        物理オブジェクトを初期化
        
        パラメータ:
        - object_id: オブジェクトのID
        - name: オブジェクトの名前
        - mass: 質量（kg）
        - position: 初期位置 [x, y, z]
        - rotation: 初期回転（四元数） [x, y, z, w]
        """
        self.id = object_id
        self.name = name
        self.mass = mass
        
        # 位置、回転、速度、角速度
        self.position = np.array(position if position else [0, 0, 0])
        self.rotation = np.array(rotation if rotation else [0, 0, 0, 1])  # 四元数 (x,y,z,w)
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        
        # 物理特性
        self.friction = 0.5
        self.restitution = 0.5  # 弾性係数
        self.linear_damping = 0.01
        self.angular_damping = 0.01
        
        # 衝突形状（単純な球体をデフォルトとする）
        self.collision_shape = {
            "type": "sphere",
            "radius": 1.0
        }
        
        # 力とトルクの蓄積値
        self.forces = np.zeros(3)
        self.torque = np.zeros(3)
        
        # オブジェクトが動的か静的か
        self.is_dynamic = True if mass > 0 else False
        
        print(f"物理オブジェクト作成: {name}（ID: {object_id}）, 質量: {mass}kg")
    
    def apply_force(self, force: List[float], local_position: List[float] = None) -> None:
        """
        オブジェクトに力を加える
        
        パラメータ:
        - force: 加える力 [x, y, z]（ワールド座標系）
        - local_position: 力を加える位置（オブジェクトのローカル座標系）
        """
        if not self.is_dynamic:
            return
        
        force = np.array(force)
        self.forces += force
        
        if local_position is not None:
            # ローカル座標を世界座標に変換（簡易実装）
            local_pos = np.array(local_position)
            # トルクの計算: トルク = 位置ベクトル × 力ベクトル
            torque = np.cross(local_pos, force)
            self.torque += torque
    
    def apply_impulse(self, impulse: List[float], local_position: List[float] = None) -> None:
        """
        オブジェクトに衝撃を加える
        
        パラメータ:
        - impulse: 加える衝撃量 [x, y, z]（ワールド座標系）
        - local_position: 衝撃を加える位置（オブジェクトのローカル座標系）
        """
        if not self.is_dynamic:
            return
        
        impulse = np.array(impulse)
        
        # 速度に直接反映
        self.velocity += impulse / self.mass
        
        if local_position is not None:
            # ローカル座標を世界座標に変換（簡易実装）
            local_pos = np.array(local_position)
            # 角運動量の計算
            angular_impulse = np.cross(local_pos, impulse)
            # 単純化のため、球形の慣性モーメントを仮定
            inertia = (2/5) * self.mass * (self.collision_shape["radius"] ** 2)
            self.angular_velocity += angular_impulse / inertia
    
    def set_position(self, position: List[float]) -> None:
        """
        オブジェクトの位置を設定
        
        パラメータ:
        - position: 新しい位置 [x, y, z]
        """
        self.position = np.array(position)
    
    def set_rotation(self, rotation: List[float]) -> None:
        """
        オブジェクトの回転を設定
        
        パラメータ:
        - rotation: 新しい回転（四元数） [x, y, z, w]
        """
        rotation = np.array(rotation)
        # 正規化
        norm = np.linalg.norm(rotation)
        if norm > 0:
            self.rotation = rotation / norm
    
    def get_state(self) -> Dict[str, Any]:
        """
        オブジェクトの現在の状態を取得
        
        戻り値:
        - 状態情報を含む辞書
        """
        return {
            "id": self.id,
            "name": self.name,
            "position": self.position.tolist(),
            "rotation": self.rotation.tolist(),
            "velocity": self.velocity.tolist(),
            "angular_velocity": self.angular_velocity.tolist(),
            "mass": self.mass,
            "is_dynamic": self.is_dynamic
        }


class Environment:
    """
    物理シミュレーション環境
    
    オブジェクトの作成、物理シミュレーションの実行、状態の取得などの機能を提供します。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        環境を初期化
        
        パラメータ:
        - config: 設定パラメータ
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # オブジェクト管理
        self.objects = {}  # ID -> オブジェクト
        self.next_object_id = 0
        
        # シミュレーション状態
        self.current_time = 0.0
        self.simulation_running = False
        self.simulation_steps = 0
        
        # 衝突検出用
        self.collisions = []  # [(object_id1, object_id2, contact_point, normal, impulse), ...]
        
        print(f"物理環境を初期化: 重力={self.config['gravity']}, タイムステップ={self.config['timestep']}s")
    
    def create_object(self, name: str = "Object", mass: float = 1.0, 
                     position: List[float] = None, rotation: List[float] = None) -> int:
        """
        新しいオブジェクトを作成
        
        パラメータ:
        - name: オブジェクトの名前
        - mass: 質量（kg）
        - position: 初期位置 [x, y, z]
        - rotation: 初期回転（四元数） [x, y, z, w]
        
        戻り値:
        - 新しいオブジェクトのID
        """
        # オブジェクト数の上限チェック
        if len(self.objects) >= self.config["max_objects"]:
            print(f"警告: オブジェクト数の上限({self.config['max_objects']})に達しました")
            return -1
        
        # 新しいオブジェクトを作成
        object_id = self.next_object_id
        self.next_object_id += 1
        
        new_object = PhysicalObject(
            object_id=object_id,
            name=name,
            mass=mass,
            position=position,
            rotation=rotation
        )
        
        # オブジェクトの登録
        self.objects[object_id] = new_object
        
        return object_id
    
    def remove_object(self, object_id: int) -> bool:
        """
        オブジェクトを環境から削除
        
        パラメータ:
        - object_id: 削除するオブジェクトのID
        
        戻り値:
        - 削除が成功した場合はTrue
        """
        if object_id in self.objects:
            obj = self.objects[object_id]
            print(f"オブジェクト削除: {obj.name}（ID: {object_id}）")
            del self.objects[object_id]
            return True
        return False
    
    def step_simulation(self, dt: float = None) -> None:
        """
        シミュレーションを1ステップ進める
        
        パラメータ:
        - dt: タイムステップ（指定しない場合は設定値を使用）
        """
        if dt is None:
            dt = self.config["timestep"]
        
        # シミュレーション状態の更新
        self.current_time += dt
        self.simulation_steps += 1
        
        # 全オブジェクトを更新
        self._update_all_objects(dt)
        
        # 衝突検出と応答
        self._detect_collisions()
        
        # デバッグ情報（100ステップごと）
        if self.simulation_steps % 100 == 0:
            print(f"シミュレーション: ステップ={self.simulation_steps}, 時間={self.current_time:.2f}s")
    
    def _update_all_objects(self, dt: float) -> None:
        """
        すべてのオブジェクトの物理状態を更新
        
        パラメータ:
        - dt: タイムステップ
        """
        gravity = np.array(self.config["gravity"])
        
        for obj in self.objects.values():
            if not obj.is_dynamic:
                continue
            
            # 重力の適用
            gravitational_force = gravity * obj.mass
            obj.forces += gravitational_force
            
            # 速度の更新（F = ma -> a = F/m）
            acceleration = obj.forces / obj.mass
            obj.velocity += acceleration * dt
            
            # 減衰の適用
            obj.velocity *= (1.0 - obj.linear_damping)
            obj.angular_velocity *= (1.0 - obj.angular_damping)
            
            # 位置の更新
            obj.position += obj.velocity * dt
            
            # 回転の更新（簡易実装 - 実際の四元数回転はもっと複雑）
            # ここでは回転の大きさが小さいと仮定して簡略化
            angular_displacement = obj.angular_velocity * dt
            quaternion_change = np.array([*angular_displacement/2, 0])
            # 四元数の掛け算（簡易実装）
            q1 = obj.rotation
            q2 = quaternion_change
            
            # 非常に簡略化した四元数の更新
            w = q1[3]*q2[3] - np.dot(q1[:3], q2[:3])
            xyz = q1[3]*q2[:3] + q2[3]*q1[:3] + np.cross(q1[:3], q2[:3])
            new_rotation = np.array([*xyz, w])
            
            # 正規化
            norm = np.linalg.norm(new_rotation)
            if norm > 0:
                obj.rotation = new_rotation / norm
            
            # 力とトルクをリセット
            obj.forces = np.zeros(3)
            obj.torque = np.zeros(3)
    
    def _detect_collisions(self) -> None:
        """
        オブジェクト間の衝突を検出し、応答を計算
        """
        self.collisions = []
        
        # すべてのオブジェクトペアで衝突をチェック
        object_ids = list(self.objects.keys())
        for i in range(len(object_ids)):
            for j in range(i + 1, len(object_ids)):
                id1, id2 = object_ids[i], object_ids[j]
                obj1, obj2 = self.objects[id1], self.objects[id2]
                
                # 両方とも静的オブジェクトなら衝突処理は不要
                if not obj1.is_dynamic and not obj2.is_dynamic:
                    continue
                
                # 簡易的な衝突判定（球体のみ）
                if obj1.collision_shape["type"] == "sphere" and obj2.collision_shape["type"] == "sphere":
                    radius1 = obj1.collision_shape["radius"]
                    radius2 = obj2.collision_shape["radius"]
                    
                    # 中心間の距離を計算
                    distance = np.linalg.norm(obj1.position - obj2.position)
                    
                    # 衝突判定
                    if distance < radius1 + radius2:
                        # 衝突方向（正規化された方向ベクトル）
                        if distance > 0:
                            normal = (obj2.position - obj1.position) / distance
                        else:
                            normal = np.array([0, 0, 1])  # デフォルトの方向
                        
                        # 衝突点（中間点として簡略化）
                        contact_point = obj1.position + normal * (distance / 2)
                        
                        # 相対速度
                        rel_velocity = obj2.velocity - obj1.velocity
                        
                        # 反発係数（弾性係数の平均）
                        restitution = (obj1.restitution + obj2.restitution) / 2
                        
                        # 衝突の強さ（簡易計算）
                        impulse_strength = max(0, np.dot(rel_velocity, normal)) * (1 + restitution)
                        impulse = normal * impulse_strength
                        
                        # 質量に応じて衝撃を分配
                        if obj1.is_dynamic and obj2.is_dynamic:
                            m1, m2 = obj1.mass, obj2.mass
                            total_mass = m1 + m2
                            obj1_ratio = m2 / total_mass
                            obj2_ratio = m1 / total_mass
                        elif obj1.is_dynamic:
                            obj1_ratio, obj2_ratio = 1.0, 0.0
                        else:
                            obj1_ratio, obj2_ratio = 0.0, 1.0
                        
                        # 衝撃の適用
                        if obj1.is_dynamic:
                            obj1.apply_impulse(impulse * obj1_ratio, contact_point - obj1.position)
                        if obj2.is_dynamic:
                            obj2.apply_impulse(-impulse * obj2_ratio, contact_point - obj2.position)
                        
                        # 衝突リストに追加
                        self.collisions.append((id1, id2, contact_point.tolist(), normal.tolist(), impulse_strength))
    
    def get_object(self, object_id: int) -> Optional[PhysicalObject]:
        """
        IDを指定してオブジェクトを取得
        
        パラメータ:
        - object_id: オブジェクトのID
        
        戻り値:
        - オブジェクト（存在しない場合はNone）
        """
        return self.objects.get(object_id)
    
    def get_all_objects(self) -> Dict[int, PhysicalObject]:
        """
        すべてのオブジェクトを取得
        
        戻り値:
        - オブジェクトの辞書（ID -> オブジェクト）
        """
        return self.objects.copy()
    
    def get_state(self) -> Dict[str, Any]:
        """
        環境の現在の状態を取得
        
        戻り値:
        - 状態情報を含む辞書
        """
        # オブジェクトの状態を取得
        object_states = {obj_id: obj.get_state() for obj_id, obj in self.objects.items()}
        
        return {
            "time": self.current_time,
            "steps": self.simulation_steps,
            "object_count": len(self.objects),
            "objects": object_states,
            "collisions": self.collisions,
            "config": self.config
        }
    
    def reset(self) -> None:
        """
        環境をリセット
        """
        self.objects = {}
        self.next_object_id = 0
        self.current_time = 0.0
        self.simulation_steps = 0
        self.collisions = []
        print("物理環境をリセット")


# 便利な関数
def create_environment(gravity: List[float] = None, timestep: float = None) -> Environment:
    """
    新しい物理環境を作成
    
    パラメータ:
    - gravity: 重力ベクトル [x, y, z]
    - timestep: シミュレーションのタイムステップ
    
    戻り値:
    - 作成された環境
    """
    config = {}
    
    if gravity is not None:
        config["gravity"] = gravity
    if timestep is not None:
        config["timestep"] = timestep
    
    return Environment(config) 