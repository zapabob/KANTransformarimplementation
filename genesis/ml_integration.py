"""
Genesis 機械学習統合モジュール

このモジュールは神経伝達物質システムとニューラルネットワークを統合し、
強化学習などの手法を用いて運動制御を最適化します。

主な機能:
- 神経伝達物質レベルの予測と最適化
- 運動パターンの学習と生成
- センサー入力からの適応学習
- 強化学習による運動制御
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import random
from collections import deque

class NeuralControlModule(nn.Module):
    """
    神経制御モジュール
    
    神経伝達物質の状態を考慮して運動制御を行うニューラルネットワーク。
    センサー入力と神経伝達物質の状態から適切な運動出力を生成します。
    """
    
    def __init__(self, 
                 nt_count: int = 8,
                 sensor_dim: int = 10,
                 motor_dim: int = 10,
                 hidden_dim: int = 128,
                 learning_rate: float = 1e-4):
        """
        神経制御モジュールを初期化
        
        パラメータ:
        - nt_count: 神経伝達物質の種類数
        - sensor_dim: センサー入力の次元数
        - motor_dim: 運動出力の次元数
        - hidden_dim: 隠れ層の次元数
        - learning_rate: 学習率
        """
        super().__init__()
        
        self.nt_count = nt_count
        self.sensor_dim = sensor_dim
        self.motor_dim = motor_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # ネットワーク層
        # センサー入力を処理するエンコーダ
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 神経伝達物質状態を処理するエンコーダ
        self.nt_encoder = nn.Sequential(
            nn.Linear(nt_count, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 結合表現から運動出力を生成するデコーダ
        self.motor_decoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, motor_dim),
            nn.Tanh()  # 運動出力を-1から1の範囲に制限
        )
        
        # 神経伝達物質の次の状態を予測するデコーダ
        self.nt_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, nt_count),
            nn.Sigmoid()  # 神経伝達物質レベルは0から1の範囲
        )
        
        # 次のセンサー入力を予測するデコーダ（予測的符号化）
        self.sensor_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2 + motor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sensor_dim)
        )
        
        # 最適化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # 損失関数
        self.motor_loss_fn = nn.MSELoss()
        self.nt_loss_fn = nn.MSELoss()
        self.prediction_loss_fn = nn.MSELoss()
        
        # 内部状態
        self.hidden_state = torch.zeros(1, hidden_dim)
        
        # 経験再生用メモリ
        self.memory_buffer = []
        self.buffer_size = 10000
        
        # 報酬の重み付け
        self.reward_weights = {
            "motor_precision": 1.0,     # 運動精度の重要度
            "energy_efficiency": 0.5,   # エネルギー効率の重要度
            "prediction_accuracy": 0.7, # 予測精度の重要度
            "nt_stability": 0.3        # 神経伝達物質の安定性の重要度
        }
        
        # トレーニングモード
        self.training = True
        
        print(f"神経制御モジュール初期化: NT数={nt_count}, センサー次元={sensor_dim}, 運動次元={motor_dim}")
    
    def forward(self, 
                sensor_input: torch.Tensor, 
                nt_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        順伝播計算
        
        パラメータ:
        - sensor_input: センサー入力 [batch_size, sensor_dim]
        - nt_state: 神経伝達物質状態 [batch_size, nt_count]
        
        戻り値:
        - motor_output: 運動出力 [batch_size, motor_dim]
        - next_nt: 予測される次の神経伝達物質状態 [batch_size, nt_count]
        - predicted_sensor: 予測される次のセンサー入力 [batch_size, sensor_dim]
        """
        # センサー入力の処理
        sensor_features = self.sensor_encoder(sensor_input)
        
        # 神経伝達物質状態の処理
        nt_features = self.nt_encoder(nt_state)
        
        # 特徴の結合
        combined_features = torch.cat([sensor_features, nt_features], dim=1)
        
        # 運動出力の生成
        motor_output = self.motor_decoder(combined_features)
        
        # 次の神経伝達物質状態の予測
        next_nt = self.nt_predictor(combined_features)
        
        # 運動出力も含めて次のセンサー入力を予測
        sensor_pred_input = torch.cat([combined_features, motor_output], dim=1)
        predicted_sensor = self.sensor_predictor(sensor_pred_input)
        
        return motor_output, next_nt, predicted_sensor
    
    def compute_reward(self, 
                       target_motor: torch.Tensor, 
                       motor_output: torch.Tensor,
                       prev_nt: torch.Tensor, 
                       next_nt: torch.Tensor,
                       sensor_input: torch.Tensor, 
                       predicted_sensor: torch.Tensor) -> torch.Tensor:
        """
        報酬を計算
        
        パラメータ:
        - target_motor: 目標運動出力
        - motor_output: 実際の運動出力
        - prev_nt: 前の神経伝達物質状態
        - next_nt: 次の神経伝達物質状態
        - sensor_input: 実際のセンサー入力
        - predicted_sensor: 予測センサー入力
        
        戻り値:
        - reward: 報酬値
        """
        # 運動精度の報酬
        motor_precision = -self.motor_loss_fn(motor_output, target_motor)
        
        # エネルギー効率の報酬（運動出力の絶対値の平均を小さくする）
        energy_efficiency = -torch.mean(torch.abs(motor_output))
        
        # 予測精度の報酬
        prediction_accuracy = -self.prediction_loss_fn(predicted_sensor, sensor_input)
        
        # 神経伝達物質の安定性報酬
        nt_stability = -torch.mean(torch.abs(next_nt - prev_nt))
        
        # 総合報酬
        reward = (
            self.reward_weights["motor_precision"] * motor_precision +
            self.reward_weights["energy_efficiency"] * energy_efficiency +
            self.reward_weights["prediction_accuracy"] * prediction_accuracy +
            self.reward_weights["nt_stability"] * nt_stability
        )
        
        return reward
    
    def train_step(self, 
                   sensor_input: torch.Tensor, 
                   nt_state: torch.Tensor, 
                   target_motor: torch.Tensor,
                   next_sensor: torch.Tensor = None,
                   alpha: float = 0.5) -> Dict[str, float]:
        """
        1ステップの訓練を実行
        
        パラメータ:
        - sensor_input: センサー入力
        - nt_state: 神経伝達物質状態
        - target_motor: 目標運動出力
        - next_sensor: 次のセンサー入力（なければ予測から学習）
        - alpha: 強化学習の重みづけ係数
        
        戻り値:
        - losses: 各種損失値の辞書
        """
        if not self.training:
            return {"motor_loss": 0.0, "nt_loss": 0.0, "prediction_loss": 0.0, "total_loss": 0.0}
        
        # 勾配をリセット
        self.optimizer.zero_grad()
        
        # 順伝播
        motor_output, next_nt, predicted_sensor = self(sensor_input, nt_state)
        
        # 運動出力の損失
        motor_loss = self.motor_loss_fn(motor_output, target_motor)
        
        # 予測損失
        if next_sensor is not None:
            prediction_loss = self.prediction_loss_fn(predicted_sensor, next_sensor)
        else:
            # 次のセンサー値が提供されない場合、予測からの学習はスキップ
            prediction_loss = torch.tensor(0.0)
        
        # 神経伝達物質の連続性損失（急激な変化を防ぐ）
        nt_loss = torch.mean(torch.abs(next_nt - nt_state)) * 0.1
        
        # 強化学習報酬
        reward = self.compute_reward(
            target_motor, motor_output, 
            nt_state, next_nt, 
            sensor_input, predicted_sensor
        )
        
        # 総合損失
        total_loss = motor_loss + prediction_loss + nt_loss - alpha * reward
        
        # 勾配計算と最適化
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "motor_loss": motor_loss.item(),
            "nt_loss": nt_loss.item(),
            "prediction_loss": prediction_loss.item(),
            "reward": reward.item(),
            "total_loss": total_loss.item()
        }
    
    def add_experience(self, 
                       sensor_input: torch.Tensor, 
                       nt_state: torch.Tensor,
                       motor_output: torch.Tensor, 
                       next_nt: torch.Tensor,
                       next_sensor: torch.Tensor, 
                       reward: float):
        """
        経験をメモリバッファに追加（経験再生用）
        """
        experience = {
            "sensor_input": sensor_input.detach(),
            "nt_state": nt_state.detach(),
            "motor_output": motor_output.detach(),
            "next_nt": next_nt.detach(),
            "next_sensor": next_sensor.detach(),
            "reward": torch.tensor(reward)
        }
        
        self.memory_buffer.append(experience)
        
        # バッファサイズ制限
        if len(self.memory_buffer) > self.buffer_size:
            self.memory_buffer.pop(0)
    
    def replay_experience(self, batch_size: int = 32, gamma: float = 0.95):
        """
        蓄積された経験からミニバッチを抽出して学習（経験再生）
        """
        if len(self.memory_buffer) < batch_size:
            return
        
        # バッファからランダムにバッチをサンプリング
        indices = np.random.choice(len(self.memory_buffer), batch_size, replace=False)
        batch = [self.memory_buffer[i] for i in indices]
        
        # バッチデータの準備
        sensor_inputs = torch.cat([exp["sensor_input"] for exp in batch], dim=0)
        nt_states = torch.cat([exp["nt_state"] for exp in batch], dim=0)
        motor_outputs = torch.cat([exp["motor_output"] for exp in batch], dim=0)
        next_nt_states = torch.cat([exp["next_nt"] for exp in batch], dim=0)
        next_sensors = torch.cat([exp["next_sensor"] for exp in batch], dim=0)
        rewards = torch.stack([exp["reward"] for exp in batch])
        
        # 現在の価値を計算
        with torch.no_grad():
            _, _, predicted_next_sensors = self(sensor_inputs, nt_states)
        
        # TD誤差を使った学習
        prediction_loss = self.prediction_loss_fn(predicted_next_sensors, next_sensors)
        temporal_diff = rewards + gamma * prediction_loss
        
        # 勾配をリセット
        self.optimizer.zero_grad()
        
        # 損失を計算
        loss = temporal_diff.mean()
        
        # 勾配計算と最適化
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, path: str):
        """モデルの保存"""
        model_data = {
            "model_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "hidden_state": self.hidden_state,
            "config": {
                "nt_count": self.nt_count,
                "sensor_dim": self.sensor_dim,
                "motor_dim": self.motor_dim,
                "hidden_dim": self.hidden_dim,
                "learning_rate": self.learning_rate,
                "reward_weights": self.reward_weights
            }
        }
        
        torch.save(model_data, path)
    
    def load_model(self, path: str):
        """モデルの読み込み"""
        model_data = torch.load(path)
        
        # 設定の検証
        config = model_data["config"]
        if (config["nt_count"] != self.nt_count or
            config["sensor_dim"] != self.sensor_dim or
            config["motor_dim"] != self.motor_dim or
            config["hidden_dim"] != self.hidden_dim):
            raise ValueError("モデル構成が一致しません")
        
        # モデルの状態を復元
        self.load_state_dict(model_data["model_state"])
        self.optimizer.load_state_dict(model_data["optimizer_state"])
        self.hidden_state = model_data["hidden_state"]
        self.learning_rate = config["learning_rate"]
        self.reward_weights = config["reward_weights"]


class ReceptorFeedbackRLSystem:
    """
    強化学習による受容体フィードバック制御システム
    
    受容体感度の適応的な調整を学習するためのRLシステム。
    """
    
    def __init__(self, nt_count: int = 8, learning_rate: float = 0.001):
        """
        受容体フィードバック強化学習システムを初期化
        
        パラメータ:
        - nt_count: 神経伝達物質の種類数
        - learning_rate: 学習率
        """
        self.nt_count = nt_count
        self.learning_rate = learning_rate
        
        # 神経伝達物質レベル
        self.nt_levels = np.ones(nt_count) * 0.5
        
        # 受容体感度
        self.sensitivity = np.ones(nt_count)
        
        # Q学習パラメータ
        self.q_table = {}  # 状態-行動値関数
        self.epsilon = 0.2  # ε-greedy探索確率
        self.gamma = 0.95  # 割引率
        self.alpha = learning_rate  # 学習率
        
        # 状態・行動の離散化
        self.level_bins = np.linspace(0, 1, 10)  # レベルを10段階に離散化
        self.sensitivity_bins = np.linspace(0.1, 3.0, 10)  # 感度を10段階に離散化
        
        # 過去の状態と報酬
        self.last_state = None
        self.last_action = None
        
        print(f"受容体フィードバックRL初期化: NT数={nt_count}")
    
    def discretize_state(self, nt_levels: np.ndarray) -> tuple:
        """
        神経伝達物質レベルを離散的な状態に変換
        
        パラメータ:
        - nt_levels: 神経伝達物質レベル
        
        戻り値:
        - 離散化された状態（タプル）
        """
        # 各レベルをビンに割り当て
        discrete_levels = tuple(np.digitize(nt_levels, self.level_bins))
        return discrete_levels
    
    def get_q_value(self, state: tuple, action: int) -> float:
        """
        Q値を取得
        
        パラメータ:
        - state: 状態
        - action: 行動
        
        戻り値:
        - Q値
        """
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0
        return self.q_table[(state, action)]
    
    def update_q_value(self, state: tuple, action: int, reward: float, next_state: tuple):
        """
        Q値を更新
        
        パラメータ:
        - state: 現在の状態
        - action: 実行した行動
        - reward: 得られた報酬
        - next_state: 次の状態
        """
        # 次の状態での最大Q値を取得
        next_q_max = max([self.get_q_value(next_state, a) for a in range(self.nt_count * 2)])
        
        # Q値の更新
        old_q = self.get_q_value(state, action)
        self.q_table[(state, action)] = old_q + self.alpha * (reward + self.gamma * next_q_max - old_q)
    
    def select_action(self, state: tuple) -> Tuple[int, str]:
        """
        状態に基づいて行動を選択
        
        パラメータ:
        - state: 現在の状態
        
        戻り値:
        - action: 選択された行動
        - action_type: 行動タイプ（"increase" または "decrease"）
        """
        # ε-greedy方策
        if np.random.random() < self.epsilon:
            # ランダムに行動を選択
            action = np.random.randint(0, self.nt_count * 2)
        else:
            # 最大Q値の行動を選択
            q_values = [self.get_q_value(state, a) for a in range(self.nt_count * 2)]
            action = np.argmax(q_values)
        
        # 行動タイプの決定
        nt_idx = action % self.nt_count
        action_type = "increase" if action < self.nt_count else "decrease"
        
        return action, f"{action_type}_{self.nt_names[nt_idx]}"
    
    def compute_reward(self, 
                     nt_levels: np.ndarray, 
                     effective_levels: np.ndarray, 
                     target_levels: np.ndarray = None) -> float:
        """
        報酬を計算
        
        パラメータ:
        - nt_levels: 実際の神経伝達物質レベル
        - effective_levels: 実効レベル（受容体感度調整後）
        - target_levels: 目標レベル（指定されていれば使用）
        
        戻り値:
        - reward: 報酬値
        """
        # 目標レベルが指定されていない場合は最適レベル（0.5）を使用
        if target_levels is None:
            target_levels = np.ones_like(nt_levels) * 0.5
        
        # 1. 実効レベルと目標レベルの差（小さいほど良い）
        level_error = np.mean(np.abs(effective_levels - target_levels))
        level_reward = 1.0 - level_error
        
        # 2. 感度の均一性（感度が極端に偏っていないほど良い）
        sensitivity_variance = np.var(self.sensitivity)
        sensitivity_reward = np.exp(-sensitivity_variance)
        
        # 3. ホメオスタシス報酬（実効レベルの安定性）
        homeostasis_reward = np.exp(-np.var(effective_levels))
        
        # 総合報酬
        reward = 0.5 * level_reward + 0.3 * sensitivity_reward + 0.2 * homeostasis_reward
        
        return reward
    
    def update(self, 
              nt_levels: np.ndarray, 
              dt: float = 0.1, 
              target_levels: np.ndarray = None) -> np.ndarray:
        """
        受容体感度を更新
        
        パラメータ:
        - nt_levels: 神経伝達物質レベル
        - dt: 時間ステップ
        - target_levels: 目標レベル
        
        戻り値:
        - effective_levels: 実効的なレベル
        """
        # 現在の状態を離散化
        current_state = self.discretize_state(nt_levels)
        
        # 行動を選択
        action, action_desc = self.select_action(current_state)
        
        # 行動を実行
        nt_idx = action % self.nt_count
        if action < self.nt_count:
            # 感度を増加
            self.sensitivity[nt_idx] = min(3.0, self.sensitivity[nt_idx] + 0.1 * dt)
        else:
            # 感度を減少
            self.sensitivity[nt_idx] = max(0.1, self.sensitivity[nt_idx] - 0.1 * dt)
        
        # 実効レベルを計算
        effective_levels = nt_levels * self.sensitivity
        effective_levels = np.clip(effective_levels, 0.0, 1.0)
        
        # 報酬を計算
        reward = self.compute_reward(nt_levels, effective_levels, target_levels)
        
        # Q値を更新（過去の状態と行動がある場合）
        if self.last_state is not None and self.last_action is not None:
            self.update_q_value(self.last_state, self.last_action, reward, current_state)
        
        # 現在の状態と行動を記録
        self.last_state = current_state
        self.last_action = action
        
        # ε値の減衰（時間とともに探索が減少）
        self.epsilon = max(0.01, self.epsilon * 0.999)
        
        return effective_levels
    
    def get_state(self) -> Dict[str, Any]:
        """現在の状態を取得"""
        return {
            "sensitivity": self.sensitivity.copy(),
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table)
        }
    
    def save_model(self, path: str):
        """モデルを保存"""
        model_data = {
            "q_table": self.q_table,
            "sensitivity": self.sensitivity,
            "epsilon": self.epsilon,
            "nt_count": self.nt_count,
            "learning_rate": self.learning_rate
        }
        
        np.save(path, model_data)
    
    def load_model(self, path: str):
        """モデルを読み込み"""
        model_data = np.load(path, allow_pickle=True).item()
        
        self.q_table = model_data["q_table"]
        self.sensitivity = model_data["sensitivity"]
        self.epsilon = model_data["epsilon"]
        self.nt_count = model_data["nt_count"]
        self.learning_rate = model_data["learning_rate"]


# ユーティリティ関数
def create_neural_control_module(config: Dict[str, Any] = None) -> NeuralControlModule:
    """
    神経制御モジュールの作成
    
    パラメータ:
    - config: 設定パラメータ
    
    戻り値:
    - 神経制御モジュール
    """
    default_config = {
        "nt_count": 8,
        "sensor_dim": 10,
        "motor_dim": 10,
        "hidden_dim": 128,
        "learning_rate": 1e-4
    }
    
    if config:
        default_config.update(config)
    
    return NeuralControlModule(**default_config) 