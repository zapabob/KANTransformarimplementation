"""
Genesis 神経伝達物質モジュール

このモジュールは複数の神経伝達物質の相互作用と動態をシミュレーションします。
脳内の神経調節メカニズムを模倣し、機械学習モデルとの統合を容易にします。

主な機能:
- 複数の神経伝達物質（ドーパミン、セロトニンなど）の動態モデル
- 神経伝達物質間の相互作用のシミュレーション
- 外部入力に対する応答特性
- 運動出力への影響の計算
- 感情状態のモデリング
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt

class NeurotransmitterSystem:
    """
    複数の神経伝達物質とその相互作用をモデル化するクラス
    
    以下の神経伝達物質の相互作用を扱います:
    - アセチルコリン: 運動制御と記憶
    - ドーパミン: 報酬と動機付け
    - セロトニン: 感情の調節と学習率
    - ノルアドレナリン: 注意と覚醒
    - グルタミン酸: 興奮性信号
    - GABA: 抑制性信号
    - エンドルフィン: 快感と痛みの抑制
    - オキシトシン: 社会的絆と信頼
    """
    
    def __init__(self, initial_levels: Dict[str, float] = None):
        """
        神経伝達物質システムを初期化
        
        パラメータ:
        - initial_levels: 神経伝達物質の初期レベル
        """
        # 初期レベル（デフォルトは0.5）
        self.levels = {
            "acetylcholine": 0.5,  # アセチルコリン
            "dopamine": 0.5,       # ドーパミン
            "serotonin": 0.5,      # セロトニン
            "noradrenaline": 0.5,  # ノルアドレナリン
            "glutamate": 0.5,      # グルタミン酸
            "gaba": 0.5,           # GABA
            "endorphin": 0.5,      # エンドルフィン
            "oxytocin": 0.5        # オキシトシン
        }
        
        # 初期値を上書き
        if initial_levels:
            for nt, level in initial_levels.items():
                if nt in self.levels:
                    self.levels[nt] = max(0.0, min(1.0, level))
        
        # 相互作用行列
        # 正の値: 増加作用、負の値: 抑制作用
        self.interaction_matrix = {
            # 行が源、列が対象
            # [アセチルコリン, ドーパミン, セロトニン, ノルアドレナリン, グルタミン酸, GABA, エンドルフィン, オキシトシン]
            "acetylcholine": [0.0, 0.02, 0.01, 0.03, 0.1, -0.05, 0.0, 0.0],
            "dopamine": [0.01, 0.0, -0.05, 0.05, 0.05, -0.03, 0.02, 0.03],
            "serotonin": [0.02, -0.04, 0.0, -0.03, -0.02, 0.05, 0.03, 0.05],
            "noradrenaline": [0.05, 0.05, -0.02, 0.0, 0.03, -0.02, -0.01, 0.0],
            "glutamate": [0.05, 0.02, -0.01, 0.03, 0.0, 0.1, 0.0, 0.01],
            "gaba": [-0.03, -0.03, 0.05, -0.05, -0.1, 0.0, 0.02, 0.03],
            "endorphin": [0.0, 0.05, 0.05, -0.02, -0.03, 0.02, 0.0, 0.05],
            "oxytocin": [0.01, 0.03, 0.05, 0.0, 0.0, 0.03, 0.05, 0.0]
        }
        
        # 自己調節係数（恒常性維持のための自己抑制）
        self.self_regulation = {
            "acetylcholine": 0.1,
            "dopamine": 0.15,
            "serotonin": 0.05,
            "noradrenaline": 0.12,
            "glutamate": 0.2,
            "gaba": 0.1,
            "endorphin": 0.08,
            "oxytocin": 0.05
        }
        
        # 減衰率（時間経過による自然減少）
        self.decay_rates = {
            "acetylcholine": 0.05,
            "dopamine": 0.1,
            "serotonin": 0.03,
            "noradrenaline": 0.08,
            "glutamate": 0.15,
            "gaba": 0.1,
            "endorphin": 0.05,
            "oxytocin": 0.03
        }
        
        # 履歴の保存
        self.history = {nt: [] for nt in self.levels.keys()}
        self.time_points = []
        self.current_time = 0.0
        
        # 最大履歴サイズ
        self.max_history = 1000
        
        print("神経伝達物質システム初期化完了")
    
    def update(self, dt: float, external_inputs: Dict[str, float] = None) -> Dict[str, float]:
        """
        神経伝達物質レベルを更新
        
        パラメータ:
        - dt: 時間ステップ
        - external_inputs: 外部入力（神経伝達物質への刺激）
        
        戻り値:
        - 更新された神経伝達物質レベル
        """
        # 時間を更新
        self.current_time += dt
        
        # 外部入力がなければ空の辞書に
        if external_inputs is None:
            external_inputs = {}
        
        # 各神経伝達物質の新しいレベルを計算
        new_levels = self.levels.copy()
        
        for nt in self.levels:
            # 現在のレベル
            current_level = self.levels[nt]
            
            # 1. 相互作用の影響を計算
            interaction_effect = 0.0
            nt_list = list(self.levels.keys())
            for i, other_nt in enumerate(nt_list):
                if other_nt != nt:
                    # 他の神経伝達物質からの影響
                    effect = self.interaction_matrix[other_nt][nt_list.index(nt)] * self.levels[other_nt]
                    interaction_effect += effect
            
            # 2. 自己調節の影響（現在のレベルに応じた自己抑制）
            self_reg_effect = -self.self_regulation[nt] * (current_level - 0.5) * 2.0
            
            # 3. 自然減衰
            decay = -self.decay_rates[nt] * current_level * dt
            
            # 4. 外部入力
            external_input = external_inputs.get(nt, 0.0) * dt
            
            # 合計変化量の計算
            total_change = (interaction_effect + self_reg_effect) * dt + decay + external_input
            
            # 新しいレベルの計算（0.0～1.0の範囲に制限）
            new_levels[nt] = max(0.0, min(1.0, current_level + total_change))
        
        # レベルの更新
        self.levels = new_levels
        
        # 履歴の更新
        self.time_points.append(self.current_time)
        for nt in self.levels:
            self.history[nt].append(self.levels[nt])
        
        # 履歴サイズの制限
        if len(self.time_points) > self.max_history:
            self.time_points.pop(0)
            for nt in self.history:
                if len(self.history[nt]) > self.max_history:
                    self.history[nt].pop(0)
        
        return self.levels
    
    def get_motor_influence(self) -> Dict[str, float]:
        """
        運動制御への影響を計算
        
        戻り値:
        - 運動パラメータ（精度、力、応答速度、持続時間）
        """
        # アセチルコリンは運動出力に直接影響
        # ドーパミンは運動の開始と維持に関与
        # ノルアドレナリンは反応速度に影響
        # セロトニンは運動リズムに関与
        # GABAは運動の抑制に関与
        
        # 精度: アセチルコリン(+)、ノルアドレナリン(-)
        precision = 0.3 + 0.5 * self.levels["acetylcholine"] - 0.2 * self.levels["noradrenaline"]
        
        # 力: ドーパミン(+)、セロトニン(-)
        strength = 0.2 + 0.6 * self.levels["dopamine"] - 0.1 * self.levels["serotonin"]
        
        # 応答速度: ノルアドレナリン(+)、GABA(-)
        response_speed = 0.3 + 0.5 * self.levels["noradrenaline"] - 0.3 * self.levels["gaba"]
        
        # 持続時間: アセチルコリン(+)、ドーパミン(+)
        duration = 0.3 + 0.3 * self.levels["acetylcholine"] + 0.4 * self.levels["dopamine"]
        
        # 各パラメータを0.0～1.0の範囲に制限
        return {
            "precision": max(0.0, min(1.0, precision)),
            "strength": max(0.0, min(1.0, strength)),
            "response_speed": max(0.0, min(1.0, response_speed)),
            "duration": max(0.0, min(1.0, duration))
        }
    
    def get_emotional_state(self) -> Dict[str, float]:
        """
        現在の感情状態を計算
        
        戻り値:
        - 感情パラメータ（快感、覚醒度、安心感、集中度）
        """
        # 快感/幸福: ドーパミン(+)、エンドルフィン(+)、セロトニン(+)
        pleasure = 0.2 * self.levels["dopamine"] + 0.4 * self.levels["endorphin"] + 0.2 * self.levels["serotonin"]
        
        # 覚醒度: ノルアドレナリン(+)、アセチルコリン(+)、グルタミン酸(+)、GABA(-)
        arousal = 0.3 * self.levels["noradrenaline"] + 0.2 * self.levels["acetylcholine"] + \
                 0.3 * self.levels["glutamate"] - 0.4 * self.levels["gaba"]
        
        # 安心感: セロトニン(+)、オキシトシン(+)、GABA(+)、ノルアドレナリン(-)
        calmness = 0.3 * self.levels["serotonin"] + 0.4 * self.levels["oxytocin"] + \
                  0.2 * self.levels["gaba"] - 0.3 * self.levels["noradrenaline"]
        
        # 集中度: ノルアドレナリン(+)、アセチルコリン(+)、ドーパミン(+)
        focus = 0.4 * self.levels["noradrenaline"] + 0.3 * self.levels["acetylcholine"] + 0.2 * self.levels["dopamine"]
        
        # 各パラメータを0.0～1.0の範囲に制限
        emotions = {
            "pleasure": max(0.0, min(1.0, pleasure)),
            "arousal": max(0.0, min(1.0, arousal)),
            "calmness": max(0.0, min(1.0, calmness)),
            "focus": max(0.0, min(1.0, focus))
        }
        
        return emotions
    
    def simulate_medication(self, medication_type: str, strength: float = 1.0) -> None:
        """
        薬物の効果をシミュレーション
        
        パラメータ:
        - medication_type: 薬物の種類
        - strength: 効果の強さ（0.0～1.0）
        """
        # 様々な薬物タイプとその効果
        medications = {
            "ssri": {  # 選択的セロトニン再取り込み阻害薬
                "serotonin": 0.3,
                "noradrenaline": 0.1
            },
            "stimulant": {  # 中枢神経刺激薬
                "dopamine": 0.4,
                "noradrenaline": 0.3,
                "serotonin": -0.1
            },
            "benzodiazepine": {  # 抗不安薬
                "gaba": 0.5,
                "glutamate": -0.2
            },
            "opioid": {  # オピオイド鎮痛薬
                "endorphin": 0.6,
                "dopamine": 0.2
            },
            "anticholinergic": {  # 抗コリン薬
                "acetylcholine": -0.5
            },
            "antipsychotic": {  # 抗精神病薬
                "dopamine": -0.4,
                "serotonin": 0.2
            }
        }
        
        if medication_type in medications:
            effects = medications[medication_type]
            
            # 効果の適用
            for nt, effect in effects.items():
                # 効果の大きさを調整
                adjusted_effect = effect * strength
                
                # 現在のレベルに効果を適用
                current = self.levels[nt]
                if effect > 0:
                    # 増加効果の場合、最大値1.0まで
                    self.levels[nt] = min(1.0, current + adjusted_effect)
                else:
                    # 減少効果の場合、最小値0.0まで
                    self.levels[nt] = max(0.0, current + adjusted_effect)
            
            print(f"{medication_type}の効果をシミュレーション（強さ: {strength}）")
        else:
            print(f"未知の薬物タイプ: {medication_type}")
    
    def plot_history(self, save_path: str = None):
        """
        神経伝達物質レベルの履歴をプロット
        
        パラメータ:
        - save_path: 保存先パス（指定があれば保存）
        """
        if not self.time_points:
            print("データがありません")
            return
        
        plt.figure(figsize=(12, 8))
        
        for nt, history in self.history.items():
            if len(history) == len(self.time_points):
                plt.plot(self.time_points, history, label=nt)
        
        plt.xlabel('時間')
        plt.ylabel('レベル')
        plt.title('神経伝達物質レベルの時間変化')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"グラフを保存しました: {save_path}")
        
        plt.show()
    
    def reset(self, initial_levels: Dict[str, float] = None):
        """
        システムをリセット
        
        パラメータ:
        - initial_levels: リセット後の初期レベル
        """
        # デフォルトの初期値に戻す
        default_levels = {
            "acetylcholine": 0.5,
            "dopamine": 0.5,
            "serotonin": 0.5,
            "noradrenaline": 0.5,
            "glutamate": 0.5,
            "gaba": 0.5,
            "endorphin": 0.5,
            "oxytocin": 0.5
        }
        
        # 初期値を上書き
        if initial_levels:
            for nt, level in initial_levels.items():
                if nt in default_levels:
                    default_levels[nt] = max(0.0, min(1.0, level))
        
        # レベルの更新
        self.levels = default_levels
        
        # 履歴のクリア
        self.history = {nt: [] for nt in self.levels.keys()}
        self.time_points = []
        self.current_time = 0.0
        
        print("神経伝達物質システムをリセットしました") 