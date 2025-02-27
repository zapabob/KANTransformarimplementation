"""
Genesis 統合システムモジュール

このモジュールは神経伝達物質、受容体、可視化、機械学習）を統合した総合システムを実装します。

主な機能:
- 複数の神経伝達物質の連携動作
- 受容体フィードバックによる適応制御
- 機械学習による運動制御最適化
- リアルタイム可視化と状態監視
"""

import numpy as np
import torch
import time
import threading
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional, Union, Any

from genesis.neurotransmitters import NeurotransmitterSystem
from genesis.receptors import ReceptorFeedbackSystem, ReceptorGatingSystem
from genesis.visualizer import NeuralVisualizer
from genesis.ml_integration import NeuralControlModule
from genesis.motor import Controller as MotorController

class IntegratedBiomimeticSystem:
    """
    包括的な生体模倣システム
    
    神経伝達物質、受容体フィードバック、機械学習、可視化機能を統合した
    完全なシミュレーションシステムです。
    """
    
    def __init__(self, 
                 sensor_dim: int = 10,
                 motor_dim: int = 10,
                 nt_names: List[str] = None,
                 visualization: bool = True):
        """
        統合システムを初期化
        
        パラメータ:
        - sensor_dim: センサー入力の次元数
        - motor_dim: 運動出力の次元数
        - nt_names: 神経伝達物質の名前リスト
        - visualization: 可視化を有効にするかどうか
        """
        # デフォルトの神経伝達物質名
        if nt_names is None:
            self.nt_names = [
                "acetylcholine", "dopamine", "serotonin", "noradrenaline",
                "glutamate", "gaba", "endorphin", "oxytocin"
            ]
        else:
            self.nt_names = nt_names
        
        self.nt_count = len(self.nt_names)
        self.sensor_dim = sensor_dim
        self.motor_dim = motor_dim
        
        # 神経伝達物質システム
        self.nt_system = NeurotransmitterSystem()
        
        # 受容体フィードバックシステム
        self.receptor_system = ReceptorFeedbackSystem(nt_names=self.nt_names)
        
        # 受容体ゲーティングシステム
        self.gating_system = ReceptorGatingSystem(nt_names=self.nt_names)
        
        # 運動制御モジュール
        self.motor_controller = MotorController(num_motors=motor_dim)
        
        # 機械学習モジュール
        self.neural_control = NeuralControlModule(
            nt_count=self.nt_count,
            sensor_dim=sensor_dim,
            motor_dim=motor_dim,
            hidden_dim=128
        )
        
        # センサー値
        self.current_sensor = np.zeros(sensor_dim)
        self.previous_sensor = np.zeros(sensor_dim)
        
        # 運動出力
        self.current_motor = np.zeros(motor_dim)
        self.target_motor = np.zeros(motor_dim)
        
        # 可視化
        self.visualization = visualization
        if visualization:
            self.visualizer = NeuralVisualizer(update_interval=100)
            self._setup_visualization()
        else:
            self.visualizer = None
        
        # 更新間隔（ミリ秒）
        self.update_interval = 50  # 20Hz
        
        # 実行スレッド
        self.running = False
        self.update_thread = None
        
        # 時間追跡
        self.start_time = None
        self.current_time = 0.0
        self.dt = 0.05  # 50ms
        
        # 内部状態記録
        self.state_history = []
        self.max_history = 1000
        
        print(f"統合バイオミメティックシステム初期化完了: NT数={self.nt_count}, 運動次元={motor_dim}")
    
    def _setup_visualization(self):
        """可視化設定"""
        # 神経伝達物質プロット
        self.visualizer.setup_neurotransmitter_plot(
            lambda: self.nt_system.levels
        )
        
        # 受容体感度プロット
        self.visualizer.setup_receptor_sensitivity_plot(
            lambda: self.receptor_system.get_all_sensitivities()
        )
        
        # 運動出力プロット
        self.visualizer.setup_motor_output_plot(
            lambda: self.motor_controller.get_state()["position"]
        )
        
        # 脳活動プロット（ダミーデータ）
        dummy_brain = np.zeros((10, 10))
        self.visualizer.setup_brain_activity_plot(
            lambda: dummy_brain + np.random.normal(0, 0.1, dummy_brain.shape) * 0.3 + 
                   np.outer(np.array([self.nt_system.levels[nt] for nt in self.nt_names]), 
                           np.ones(10)) * 0.7
        )
        
        # ダッシュボード作成
        self.visualizer.create_dashboard()
    
    def set_sensor_input(self, sensor_values: np.ndarray):
        """センサー入力を設定"""
        if len(sensor_values) != self.sensor_dim:
            raise ValueError(f"センサー値の次元数が一致しません: 期待={self.sensor_dim}, 受信={len(sensor_values)}")
        
        self.previous_sensor = self.current_sensor.copy()
        self.current_sensor = sensor_values
    
    def set_target_motor(self, motor_values: np.ndarray):
        """目標運動出力を設定"""
        if len(motor_values) != self.motor_dim:
            raise ValueError(f"運動値の次元数が一致しません: 期待={self.motor_dim}, 受信={len(motor_values)}")
        
        self.target_motor = motor_values
    
    def update(self, dt: float = None):
        """
        システム全体を更新
        
        パラメータ:
        - dt: 時間ステップ（指定がなければデフォルト値を使用）
        """
        if dt is None:
            dt = self.dt
        
        self.current_time += dt
        
        # 1. 神経伝達物質レベルの更新
        
        # センサー活動やモーター活動からの外部入力を計算
        sensor_activity = np.mean(np.abs(self.current_sensor))
        motor_activity = np.mean(np.abs(self.current_motor))
        
        # センサー入力の変化量（ノルアドレナリンに影響）
        sensor_change = np.mean(np.abs(self.current_sensor - self.previous_sensor))
        
        # 外部入力辞書の作成
        external_inputs = {
            "acetylcholine": motor_activity * 0.5,  # 運動活動に応じてACh増加
            "dopamine": sensor_activity * 0.3 if sensor_activity > 0.5 else 0,  # 高いセンサー活動でドーパミン増加
            "noradrenaline": sensor_change * 2.0,  # センサー変化でノルアドレナリン増加
            "glutamate": sensor_activity * 0.7,  # センサー活動に応じてグルタミン酸増加
            "gaba": (1.0 - sensor_activity) * 0.3  # 低センサー活動でGABA増加（抑制）
        }
        
        # 神経伝達物質の更新
        nt_levels = self.nt_system.update(dt, external_inputs)
        
        # 2. 受容体フィードバックの更新
        effective_levels = self.receptor_system.update(nt_levels, dt)
        
        # 3. 受容体ゲーティングの更新
        channel_signals = self.gating_system.update(effective_levels, dt)
        
        # 4. 神経制御モジュールによる運動出力の生成
        
        # センサー入力とNTレベルをテンソルに変換
        sensor_tensor = torch.tensor(self.current_sensor, dtype=torch.float32).unsqueeze(0)
        nt_tensor = torch.tensor(
            [effective_levels[nt] for nt in self.nt_names], 
            dtype=torch.float32
        ).unsqueeze(0)
        
        # 運動出力の生成
        with torch.no_grad():  # 推論モードで実行
            motor_output, next_nt, predicted_sensor = self.neural_control(
                sensor_tensor, nt_tensor
            )
        
        # テンソルからNumPy配列に変換
        motor_np = motor_output.squeeze(0).numpy()
        self.current_motor = motor_np
        
        # 5. 運動コントローラーの更新
        
        # アセチルコリンレベルを取得
        ach_level = self.nt_system.levels["acetylcholine"]
        
        # 運動コマンドの設定
        self.motor_controller.set_motor_command(
            motor_np, 
            acetylcholine_level=ach_level
        )
        
        # 運動コントローラーの更新
        self.motor_controller.update(dt)
        
        # 6. 学習（トレーニングモードの場合）
        if hasattr(self.neural_control, 'training') and self.neural_control.training:
            target_tensor = torch.tensor(self.target_motor, dtype=torch.float32).unsqueeze(0)
            next_sensor_tensor = torch.tensor(self.current_sensor, dtype=torch.float32).unsqueeze(0)
            
            # 1ステップの訓練
            losses = self.neural_control.train_step(
                sensor_tensor, nt_tensor, target_tensor, next_sensor_tensor
            )
            
            # 経験を記録
            reward = np.mean(1.0 - np.abs(motor_np - self.target_motor))
            
            self.neural_control.add_experience(
                sensor_tensor, nt_tensor, motor_output,
                next_nt, next_sensor_tensor, reward
            )
            
            # 定期的に経験再生を実行
            if np.random.random() < 0.1:  # 10%の確率で
                self.neural_control.replay_experience(batch_size=32)
        
        # 7. 状態の記録
        state = {
            "time": self.current_time,
            "nt_levels": {name: self.nt_system.levels[name] for name in self.nt_names},
            "effective_levels": {name: effective_levels.get(name, 0.0) for name in self.nt_names},
            "receptor_sensitivities": self.receptor_system.get_all_sensitivities(),
            "motor_output": self.current_motor.tolist(),
            "target_motor": self.target_motor.tolist(),
            "sensor_input": self.current_sensor.tolist()
        }
        
        self.state_history.append(state)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
    
    def _update_loop(self):
        """更新ループ（バックグラウンドスレッド）"""
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # システム更新
            self.update(dt)
            
            # 処理時間を考慮した待機
            elapsed = time.time() - current_time
            wait_time = max(0, self.update_interval / 1000 - elapsed)
            if wait_time > 0:
                time.sleep(wait_time)
    
    def start(self):
        """システムの実行を開始"""
        self.running = True
        self.start_time = time.time()
        
        # 更新スレッドを開始
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # 可視化を開始（設定されている場合）
        if self.visualization and self.visualizer is not None:
            self.visualizer.start_visualization()
        
        print("統合システムを開始しました")
    
    def stop(self):
        """システムの実行を停止"""
        self.running = False
        
        # 更新スレッドの終了を待機
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        
        # 可視化の停止
        if self.visualization and self.visualizer is not None:
            self.visualizer.stop_visualization()
        
        # モーターコントローラーの停止
        self.motor_controller.stop()
        
        print("統合システムを停止しました")
    
    def save_state(self, filename: str = None):
        """現在の状態を保存"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"biomimetic_state_{timestamp}.npz"
        
        # 各モジュールの状態を保存
        np.savez(
            filename,
            nt_levels=np.array([self.nt_system.levels[nt] for nt in self.nt_names]),
            receptor_sensitivity=np.array([self.receptor_system.sensitivity[nt] for nt in self.nt_names]),
            current_sensor=self.current_sensor,
            current_motor=self.current_motor,
            target_motor=self.target_motor,
            current_time=self.current_time
        )
        
        # ニューラルネットワークモデルを保存
        model_path = filename.replace(".npz", "_model.pt")
        self.neural_control.save_model(model_path)
        
        print(f"システム状態を保存しました: {filename}")
    
    def load_state(self, filename: str):
        """保存された状態を読み込む"""
        data = np.load(filename, allow_pickle=True)
        
        # 状態の復元
        nt_levels = data["nt_levels"]
        for i, name in enumerate(self.nt_names):
            if i < len(nt_levels):
                self.nt_system.levels[name] = nt_levels[i]
        
        receptor_sensitivity = data["receptor_sensitivity"]
        for i, name in enumerate(self.nt_names):
            if i < len(receptor_sensitivity):
                self.receptor_system.sensitivity[name] = receptor_sensitivity[i]
        
        self.current_sensor = data["current_sensor"]
        self.current_motor = data["current_motor"]
        self.target_motor = data["target_motor"]
        self.current_time = data["current_time"].item()
        
        # ニューラルネットワークモデルを読み込む
        model_path = filename.replace(".npz", "_model.pt")
        if os.path.exists(model_path):
            self.neural_control.load_model(model_path)
        
        print(f"システム状態を読み込みました: {filename}")
    
    def get_state(self) -> Dict[str, Any]:
        """現在の状態を取得"""
        return {
            "time": self.current_time,
            "nt_levels": {name: self.nt_system.levels[name] for name in self.nt_names},
            "receptor_sensitivities": self.receptor_system.get_all_sensitivities(),
            "motor_output": self.current_motor.tolist(),
            "target_motor": self.target_motor.tolist(),
            "sensor_input": self.current_sensor.tolist(),
            "running": self.running
        } 