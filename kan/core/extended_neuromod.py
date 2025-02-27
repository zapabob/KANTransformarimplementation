import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np
from collections import deque

class ExtendedNeuromodulator:
    """
    拡張された神経調節モジュール
    - 従来のドーパミン、ノルアドレナリン、セロトニンに加え
    - アセチルコリン、グルタミン酸、GABAを追加
    """
    def __init__(self, 
                 dopamine_init: float = 0.5,
                 noradrenaline_init: float = 0.5,
                 serotonin_init: float = 0.5,
                 acetylcholine_init: float = 0.5,
                 glutamate_init: float = 0.5,
                 gaba_init: float = 0.5):
        # 初期値の設定（0.0～1.0の範囲）
        self.dopamine = dopamine_init
        self.noradrenaline = noradrenaline_init
        self.serotonin = serotonin_init
        self.acetylcholine = acetylcholine_init
        self.glutamate = glutamate_init
        self.gaba = gaba_init
        
        # 変化率の初期化
        self.dopamine_rate = 0.0
        self.noradrenaline_rate = 0.0
        self.serotonin_rate = 0.0
        self.acetylcholine_rate = 0.0
        self.glutamate_rate = 0.0
        self.gaba_rate = 0.0
        
        # 履歴記録用
        self.history = {
            'dopamine': [],
            'noradrenaline': [],
            'serotonin': [],
            'acetylcholine': [], 
            'glutamate': [], 
            'gaba': []
        }
        
        # 時間ステップ
        self.time_step = 0
    
    def update(self, 
              activity: float, 
              sparsity: float, 
              stability: float, 
              motor_activity: float = 0.0,
              excitation_rate: float = 0.5,
              time_step: int = None):
        """神経調節状態の更新
        activity: ニューロン活動の平均値（ドーパミンに影響）
        sparsity: 活性化の疎密性（ノルアドレナリンに影響）
        stability: 活性化の安定性（セロトニンに影響）
        motor_activity: 運動関連活動（アセチルコリンに影響）
        excitation_rate: 興奮性信号の割合（グルタミン酸とGABAに影響）
        time_step: 現在の時間ステップ（省略可）
        """
        if time_step is not None:
            self.time_step = time_step
        else:
            self.time_step += 1
        
        # 各神経調節物質の更新率計算
        dopamine_target = min(1.0, activity * 2)  # 活動が高いほどドーパミン増加
        self.dopamine_rate = (dopamine_target - self.dopamine) * 0.1
        
        noradrenaline_target = min(1.0, sparsity * 2)  # スパーシティが高いほどノルアドレナリン増加
        self.noradrenaline_rate = (noradrenaline_target - self.noradrenaline) * 0.1
        
        serotonin_target = min(1.0, stability * 2)  # 安定性が高いほどセロトニン増加
        self.serotonin_rate = (serotonin_target - self.serotonin) * 0.1
        
        # 新しい神経調節物質の更新率計算
        acetylcholine_target = min(1.0, motor_activity * 3)  # 運動活動が高いほどアセチルコリン増加
        self.acetylcholine_rate = (acetylcholine_target - self.acetylcholine) * 0.15
        
        glutamate_target = min(1.0, excitation_rate * 1.5)  # 興奮性信号の割合に基づく
        self.glutamate_rate = (glutamate_target - self.glutamate) * 0.2
        
        gaba_target = min(1.0, (1.0 - excitation_rate) * 1.5)  # 抑制性信号の割合に基づく
        self.gaba_rate = (gaba_target - self.gaba) * 0.2
        
        # 相互作用（神経調節物質間の影響）
        # ドーパミンが高いと、グルタミン酸も増加傾向
        self.glutamate_rate += self.dopamine * 0.01
        
        # セロトニンが高いと、GABAも増加傾向（抑制作用の強化）
        self.gaba_rate += self.serotonin * 0.01
        
        # 実際の値の更新（0.0～1.0の範囲に制限）
        self.dopamine = max(0.0, min(1.0, self.dopamine + self.dopamine_rate))
        self.noradrenaline = max(0.0, min(1.0, self.noradrenaline + self.noradrenaline_rate))
        self.serotonin = max(0.0, min(1.0, self.serotonin + self.serotonin_rate))
        self.acetylcholine = max(0.0, min(1.0, self.acetylcholine + self.acetylcholine_rate))
        self.glutamate = max(0.0, min(1.0, self.glutamate + self.glutamate_rate))
        self.gaba = max(0.0, min(1.0, self.gaba + self.gaba_rate))
        
        # 履歴の更新
        self.history['dopamine'].append(self.dopamine)
        self.history['noradrenaline'].append(self.noradrenaline)
        self.history['serotonin'].append(self.serotonin)
        self.history['acetylcholine'].append(self.acetylcholine)
        self.history['glutamate'].append(self.glutamate)
        self.history['gaba'].append(self.gaba)
        
        # 履歴が長すぎる場合、古いものを削除
        max_history = 1000
        if len(self.history['dopamine']) > max_history:
            for key in self.history:
                self.history[key] = self.history[key][-max_history:]
    
    def get_state(self) -> Dict[str, float]:
        """現在の神経調節状態を取得"""
        return {
            'dopamine': self.dopamine,
            'noradrenaline': self.noradrenaline,
            'serotonin': self.serotonin,
            'acetylcholine': self.acetylcholine,
            'glutamate': self.glutamate,
            'gaba': self.gaba
        }
    
    def get_history(self) -> Dict[str, List[float]]:
        """神経調節状態の履歴を取得"""
        return self.history


class Astrocyte:
    """
    アストロサイト（星状膠細胞）シミュレーション
    - 長期的なコンテキスト維持
    - 代謝サポート
    - 多シナプス情報統合
    """
    def __init__(self, memory_size: int = 1000):
        self.memory_size = memory_size
        self.context_memory = deque(maxlen=memory_size)
        self.activity = 0.5  # 活動レベル（0.0～1.0）
        self.metabolic_support = 0.5  # 代謝サポートレベル（0.0～1.0）
        self.calcium_wave = 0.0  # カルシウム波（情報伝達）
        self.time_scale = 0.01  # アストロサイトの時間スケール（ニューロンより遅い）
        
        # 履歴
        self.history = {
            'activity': [],
            'metabolic_support': [],
            'calcium_wave': []
        }
    
    def update(self, neural_activity: np.ndarray, time_scale: float = 0.01):
        """
        アストロサイトの状態更新
        neural_activity: ニューロン活動の配列
        time_scale: 時間スケール（デフォルトは0.01、ニューロンの1/100のスピード）
        """
        # 新しいニューロン活動を記憶に追加
        if len(neural_activity.shape) > 1:
            # 多次元配列の場合、平均を取る
            mean_activity = np.mean(np.abs(neural_activity))
            self.context_memory.append(mean_activity)
        else:
            self.context_memory.append(np.mean(np.abs(neural_activity)))
        
        # 長期的活動パターンに基づく活動レベルの更新
        recent_activity = np.mean(list(self.context_memory)[-10:])
        long_term_activity = np.mean(list(self.context_memory))
        
        # 現在と長期的な活動の差に基づいてカルシウム波を生成
        calcium_target = np.abs(recent_activity - long_term_activity) * 2.0
        self.calcium_wave += (calcium_target - self.calcium_wave) * time_scale
        
        # 活動レベルの更新（カルシウム波に影響される）
        activity_target = recent_activity * (1.0 + self.calcium_wave)
        self.activity += (activity_target - self.activity) * time_scale
        self.activity = max(0.0, min(1.0, self.activity))
        
        # 代謝サポートの更新（長期活動に基づく）
        # 持続的に活動が高い領域に対してより多くの代謝サポートを提供
        metabolic_target = long_term_activity * 1.5
        self.metabolic_support += (metabolic_target - self.metabolic_support) * time_scale * 0.5
        self.metabolic_support = max(0.3, min(1.0, self.metabolic_support))  # 最低限のサポートは保証
        
        # 履歴の更新
        self.history['activity'].append(self.activity)
        self.history['metabolic_support'].append(self.metabolic_support)
        self.history['calcium_wave'].append(self.calcium_wave)
        
        # 履歴が長すぎる場合、古いものを削除
        max_history = 1000
        if len(self.history['activity']) > max_history:
            for key in self.history:
                self.history[key] = self.history[key][-max_history:]
    
    def get_context_modulation(self) -> float:
        """コンテキスト変調値の取得"""
        # 短期と長期の記憶の差に基づく変調
        if len(self.context_memory) < 10:
            return 1.0
        
        recent_memory = list(self.context_memory)[-10:]
        old_memory = list(self.context_memory)[:-10]
        
        if not old_memory:  # 古い記憶がまだ不十分な場合
            return 1.0
        
        recent_mean = np.mean(recent_memory)
        old_mean = np.mean(old_memory)
        
        # 短期と長期で大きな差がある場合、変調が大きくなる
        modulation = 1.0 + np.abs(recent_mean - old_mean) * 2.0
        return min(2.0, modulation)  # 最大2倍の変調
    
    def get_metabolic_support(self) -> float:
        """代謝サポートレベルの取得"""
        return self.metabolic_support
    
    def get_state(self) -> Dict[str, Any]:
        """アストロサイトの現在の状態を取得"""
        return {
            'activity': self.activity,
            'metabolic_support': self.metabolic_support,
            'calcium_wave': self.calcium_wave,
            'memory_length': len(self.context_memory),
            'context_modulation': self.get_context_modulation()
        }


class Microglia:
    """
    ミクログリア（小膠細胞）シミュレーション
    - ネットワークの自己修復機能
    - シナプスのプルーニング
    - 不要な結合の除去と重要な結合の強化
    """
    def __init__(self, memory_size: int = 500):
        self.memory_size = memory_size
        self.activity_memory = deque(maxlen=memory_size)
        self.repair_rate = 0.5  # 修復活性（0.0～1.0）
        self.pruning_threshold = 0.3  # プルーニング閾値
        self.enhancement_rate = 0.01  # 強化率
        
        # 履歴
        self.history = {
            'repair_rate': [],
            'pruning_threshold': [],
            'enhancement_rate': []
        }
    
    def update(self, neural_activity: np.ndarray, connection_strengths: np.ndarray):
        """
        ミクログリアの状態更新
        neural_activity: ニューロン活動の配列
        connection_strengths: シナプス結合強度の配列
        """
        # 新しいニューロン活動を記憶に追加
        if len(neural_activity.shape) > 1:
            # 多次元配列の場合、平均を取る
            mean_activity = np.mean(np.abs(neural_activity))
            self.activity_memory.append(mean_activity)
        else:
            self.activity_memory.append(np.mean(np.abs(neural_activity)))
        
        # 結合の使用状況に基づく修復率の計算
        weak_connections = np.mean(connection_strengths < self.pruning_threshold)
        strong_connections = np.mean(connection_strengths > 0.7)
        
        # 弱い結合が多すぎる場合は修復率を上げる
        if weak_connections > 0.5:
            self.repair_rate += 0.01
        else:
            self.repair_rate -= 0.005
        
        # 強い結合が多すぎる場合はプルーニングを強化
        if strong_connections > 0.7:
            self.pruning_threshold += 0.01
            self.enhancement_rate -= 0.001
        else:
            self.pruning_threshold -= 0.005
            self.enhancement_rate += 0.0005
        
        # 範囲制限
        self.repair_rate = max(0.1, min(0.9, self.repair_rate))
        self.pruning_threshold = max(0.1, min(0.5, self.pruning_threshold))
        self.enhancement_rate = max(0.001, min(0.02, self.enhancement_rate))
        
        # 履歴の更新
        self.history['repair_rate'].append(self.repair_rate)
        self.history['pruning_threshold'].append(self.pruning_threshold)
        self.history['enhancement_rate'].append(self.enhancement_rate)
        
        # 履歴が長すぎる場合、古いものを削除
        max_history = 1000
        if len(self.history['repair_rate']) > max_history:
            for key in self.history:
                self.history[key] = self.history[key][-max_history:]
    
    def get_repair_factors(self) -> Dict[str, float]:
        """修復関連の値を取得"""
        return {
            'repair_rate': self.repair_rate,
            'pruning_threshold': self.pruning_threshold,
            'enhancement_rate': self.enhancement_rate
        }
    
    def get_state(self) -> Dict[str, Any]:
        """ミクログリアの現在の状態を取得"""
        activity_stats = {
            'mean': np.mean(list(self.activity_memory)) if self.activity_memory else 0,
            'std': np.std(list(self.activity_memory)) if self.activity_memory else 0,
            'min': np.min(list(self.activity_memory)) if self.activity_memory else 0,
            'max': np.max(list(self.activity_memory)) if self.activity_memory else 0
        }
        
        return {
            'repair_rate': self.repair_rate,
            'pruning_threshold': self.pruning_threshold,
            'enhancement_rate': self.enhancement_rate,
            'activity_stats': activity_stats
        }


class AsynchronousSpiking:
    """
    非同期スパイキングニューロンモデル
    - 神経調節物質の影響を受けるスパイク生成
    - 膜電位の時間的変化を考慮
    """
    def __init__(self, num_neurons: int):
        # 膜電位
        self.membrane_potentials = np.zeros(num_neurons, dtype=np.float32)
        
        # 発火閾値（ニューロンごとに異なる）
        self.thresholds = np.random.uniform(0.3, 0.7, num_neurons).astype(np.float32)
        
        # 不応期カウンター
        self.refractory_counters = np.zeros(num_neurons, dtype=np.float32)
        
        # 最大不応期（ニューロンごとに異なる）
        self.max_refractory = np.random.randint(3, 8, num_neurons).astype(np.float32)
        
        # スパイク履歴
        self.spike_history = []
        
        # 時間ステップ
        self.time_step = 0
    
    def update(self, inputs: np.ndarray, neuromod_state: Dict[str, float], dt: float = 1.0) -> np.ndarray:
        """
        入力に基づいてスパイクを生成
        inputs: 入力信号（ニューロンごとの入力強度）
        neuromod_state: 神経調節物質の状態
        dt: 時間ステップサイズ
        """
        self.time_step += 1
        
        # 入力をfloat32型に変換
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs, dtype=np.float32)
        elif inputs.dtype != np.float32:
            inputs = inputs.astype(np.float32)
            
        num_neurons = len(self.membrane_potentials)
        spikes = np.zeros(num_neurons, dtype=np.float32)
        
        # 神経調節物質の影響を計算
        dopamine = neuromod_state.get('dopamine', 0.5)
        noradrenaline = neuromod_state.get('noradrenaline', 0.5)
        serotonin = neuromod_state.get('serotonin', 0.5)
        acetylcholine = neuromod_state.get('acetylcholine', 0.5)
        glutamate = neuromod_state.get('glutamate', 0.5)
        gaba = neuromod_state.get('gaba', 0.5)
        
        # 神経調節物質に基づく膜電位変化率の調整
        excitability = 1.0 + 0.5 * (dopamine - 0.5) + 0.3 * (acetylcholine - 0.5) + 0.4 * (glutamate - 0.5) - 0.4 * (gaba - 0.5)
        noise_scale = 0.1 + 0.2 * noradrenaline  # ノルアドレナリンはノイズレベルを増加
        adaptation_rate = 0.1 + 0.2 * serotonin  # セロトニンは適応率を増加
        
        # 各ニューロンの更新
        for i in range(num_neurons):
            # 不応期中のニューロンはスキップ
            if self.refractory_counters[i] > 0:
                self.refractory_counters[i] -= dt
                continue
                
            # 入力と神経調節物質に基づく膜電位の更新
            input_current = inputs[i] * excitability
            
            # ノイズの追加（ノルアドレナリンの影響を受ける）
            noise = float(np.random.normal(0, noise_scale))
            
            # 膜電位の更新
            self.membrane_potentials[i] += (input_current + noise) * dt
            
            # 適応（セロトニンの影響を受ける）
            if self.membrane_potentials[i] > 0:
                self.membrane_potentials[i] -= adaptation_rate * dt * self.membrane_potentials[i]
            
            # 発火閾値を超えた場合、スパイク生成
            if self.membrane_potentials[i] >= self.thresholds[i]:
                spikes[i] = 1.0
                
                # スパイク後の処理
                self.membrane_potentials[i] = 0.0  # 膜電位をリセット
                self.refractory_counters[i] = self.max_refractory[i]  # 不応期の設定
        
        # スパイク履歴の更新
        if len(self.spike_history) > 100:  # 履歴は100ステップまで保持
            self.spike_history.pop(0)
        self.spike_history.append(spikes)
        
        return spikes
    
    def get_state(self) -> Dict[str, Any]:
        """スパイキングモジュールの現在の状態を取得"""
        # 発火率の計算（最近のステップの平均）
        if len(self.spike_history) > 0:
            firing_rates = np.mean(self.spike_history, axis=0)
        else:
            firing_rates = np.zeros_like(self.membrane_potentials)
        
        num_neurons = len(self.membrane_potentials)
        
        return {
            'mean_membrane_potential': float(np.mean(self.membrane_potentials)),
            'firing_rate': float(np.mean(firing_rates)),
            'active_neurons': float(np.sum(firing_rates > 0) / num_neurons),
            'time_step': self.time_step
        } 