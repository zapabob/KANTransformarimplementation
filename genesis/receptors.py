"""
Genesis 受容体フィードバックモジュール

このモジュールは神経伝達物質の受容体動態と適応メカニズムをシミュレーションします。
受容体の脱感作、再感作、およびホメオスタシス（恒常性維持）機能を模倣します。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import time

class ReceptorFeedbackSystem:
    """
    神経伝達物質受容体のフィードバック機構をモデル化するクラス
    
    主な機能:
    - 受容体感度の動的調整
    - 脱感作と再感作のシミュレーション
    - 交差脱感作（一方の伝達物質が他の受容体に影響）
    - ホメオスタシス維持メカニズム
    - 薬理学的介入の効果モデリング
    """
    
    def __init__(self, 
                 nt_names: List[str] = None, 
                 initial_sensitivity: Dict[str, float] = None):
        """
        受容体フィードバックシステムを初期化
        
        パラメータ:
        - nt_names: 神経伝達物質の名前リスト
        - initial_sensitivity: 初期受容体感度
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
        
        # 受容体感度（初期値は1.0、神経伝達物質の効果が100%）
        self.sensitivity = {}
        for nt in self.nt_names:
            self.sensitivity[nt] = 1.0
        
        # 初期感度を上書き
        if initial_sensitivity:
            for nt, sens in initial_sensitivity.items():
                if nt in self.sensitivity:
                    self.sensitivity[nt] = max(0.1, min(3.0, sens))
        
        # 受容体パラメータ
        
        # 脱感作速度係数（高値 = 素早く脱感作）
        self.desensitization_rate = {
            "acetylcholine": 0.15,   # 中程度
            "dopamine": 0.25,        # 高速
            "serotonin": 0.1,        # 低速
            "noradrenaline": 0.2,    # 中高速
            "glutamate": 0.3,        # 高速
            "gaba": 0.15,            # 中程度
            "endorphin": 0.3,        # 高速
            "oxytocin": 0.08         # 低速
        }
        
        # 再感作速度係数（高値 = 素早く回復）
        self.resensitization_rate = {
            "acetylcholine": 0.02,   # 中程度
            "dopamine": 0.03,        # 中程度
            "serotonin": 0.01,       # 低速
            "noradrenaline": 0.04,   # 中高速
            "glutamate": 0.05,       # 高速
            "gaba": 0.03,            # 中程度
            "endorphin": 0.02,       # 中低速
            "oxytocin": 0.01         # 低速
        }
        
        # 受容体タイプ
        # 0: イオンチャネル型（速い応答、高い脱感作率）
        # 1: 代謝調節型（遅い応答、持続的効果）
        self.receptor_types = {
            "acetylcholine": [0, 1],  # 両方のタイプが存在
            "dopamine": [1, 1],       # 主に代謝調節型
            "serotonin": [1, 1],      # 主に代謝調節型
            "noradrenaline": [0, 1],  # 両方のタイプが存在
            "glutamate": [0, 1],      # 両方のタイプが存在（NMDA、AMPA、代謝型）
            "gaba": [0, 1],           # 両方のタイプが存在（GABA-A、GABA-B）
            "endorphin": [0, 1],      # μ/κ/δ受容体
            "oxytocin": [1]           # 主に代謝調節型
        }
        
        # 交差脱感作マトリックス（神経伝達物質間の相互調節）
        # 値が正：脱感作を促進、値が負：感度を高める
        self.cross_regulation = np.zeros((self.nt_count, self.nt_count))
        
        # 交差脱感作の例
        # ドーパミンはセロトニン受容体を脱感作させる
        nt_idx = {name: i for i, name in enumerate(self.nt_names)}
        if "dopamine" in nt_idx and "serotonin" in nt_idx:
            self.cross_regulation[nt_idx["dopamine"], nt_idx["serotonin"]] = 0.1
        
        # セロトニンはドーパミン受容体を脱感作させる
        if "serotonin" in nt_idx and "dopamine" in nt_idx:
            self.cross_regulation[nt_idx["serotonin"], nt_idx["dopamine"]] = 0.05
        
        # グルタミン酸はGABA受容体を脱感作させる
        if "glutamate" in nt_idx and "gaba" in nt_idx:
            self.cross_regulation[nt_idx["glutamate"], nt_idx["gaba"]] = 0.15
        
        # 最適レベル（ホメオスタシス目標値）
        self.optimal_level = {nt: 0.5 for nt in self.nt_names}
        
        # 高レベル閾値（この値以上で脱感作が始まる）
        self.high_threshold = {nt: 0.7 for nt in self.nt_names}
        
        # 低レベル閾値（この値以下で再感作が始まる）
        self.low_threshold = {nt: 0.3 for nt in self.nt_names}
        
        # 受容体感度の履歴
        self.sensitivity_history = {nt: [] for nt in self.nt_names}
        self.time_points = []
        self.current_time = 0.0
        
        # 最大履歴サイズ
        self.max_history = 1000
        
        # 受容体健全性（長期的な脱感作により低下、修復により回復）
        self.receptor_health = {nt: 1.0 for nt in self.nt_names}
        
        print("受容体フィードバックシステム初期化完了")
    
    def update(self, 
               nt_levels: Dict[str, float], 
               dt: float = 0.1) -> Dict[str, float]:
        """
        受容体感度を更新し、実効的な神経伝達物質レベルを計算
        
        パラメータ:
        - nt_levels: 神経伝達物質レベル（0.0～1.0）
        - dt: 時間ステップ
        
        戻り値:
        - effective_levels: 受容体感度による調整後の実効レベル
        """
        # 時間を更新
        self.current_time += dt
        
        # 神経伝達物質レベルの辞書から配列に変換
        nt_array = np.zeros(self.nt_count)
        for i, nt in enumerate(self.nt_names):
            if nt in nt_levels:
                nt_array[i] = nt_levels[nt]
        
        # 受容体感度を配列に変換
        sensitivity_array = np.zeros(self.nt_count)
        for i, nt in enumerate(self.nt_names):
            sensitivity_array[i] = self.sensitivity[nt]
        
        # 交差脱感作効果の計算
        cross_effects = np.zeros(self.nt_count)
        for i in range(self.nt_count):
            for j in range(self.nt_count):
                if i != j:  # 自分自身には影響しない
                    # レベルが高いほど影響が強い
                    level_effect = max(0, nt_array[j] - self.high_threshold[self.nt_names[j]])
                    cross_effects[i] += level_effect * self.cross_regulation[j, i] * dt
        
        # 各神経伝達物質の受容体感度を更新
        for i, nt in enumerate(self.nt_names):
            level = nt_array[i]
            
            # 高レベルでの脱感作
            desens_amount = 0
            if level > self.high_threshold[nt]:
                # レベルが閾値を超えた分に比例して脱感作
                overstimulation = level - self.high_threshold[nt]
                desens_amount = self.desensitization_rate[nt] * overstimulation * dt
                
                # 受容体タイプに応じた効果の調整
                # イオンチャネル型（タイプ0）は脱感作が速い
                if 0 in self.receptor_types[nt]:
                    desens_amount *= 1.5
            
            # 低レベルでの再感作
            resens_amount = 0
            if level < self.low_threshold[nt]:
                # レベルが閾値より低いほど再感作が速い
                understimulation = self.low_threshold[nt] - level
                resens_amount = self.resensitization_rate[nt] * understimulation * dt
                
                # 受容体健全性による制限（損傷があると再感作が遅い）
                resens_amount *= self.receptor_health[nt]
            
            # 交差脱感作効果の適用
            cross_effect = cross_effects[i]
            
            # 感度の更新（脱感作、再感作、交差効果）
            net_change = resens_amount - desens_amount - cross_effect
            
            # 感度限界（0.1～3.0）
            self.sensitivity[nt] = max(0.1, min(3.0, self.sensitivity[nt] + net_change))
            
            # 受容体健全性の更新（長期的な過剰刺激により低下）
            if level > 0.9 and self.sensitivity[nt] < 0.3:
                # 高レベル + 低感度 = 受容体の損傷リスク
                self.receptor_health[nt] = max(0.5, self.receptor_health[nt] - 0.01 * dt)
            else:
                # 徐々に回復
                self.receptor_health[nt] = min(1.0, self.receptor_health[nt] + 0.001 * dt)
        
        # 実効的なレベルの計算（レベル × 感度）
        effective_levels = {}
        for i, nt in enumerate(self.nt_names):
            if nt in nt_levels:
                effective_levels[nt] = min(1.0, nt_levels[nt] * self.sensitivity[nt])
        
        # 履歴を更新
        self.time_points.append(self.current_time)
        for nt in self.nt_names:
            self.sensitivity_history[nt].append(self.sensitivity[nt])
        
        # 履歴サイズの制限
        if len(self.time_points) > self.max_history:
            self.time_points.pop(0)
            for nt in self.sensitivity_history:
                if len(self.sensitivity_history[nt]) > self.max_history:
                    self.sensitivity_history[nt].pop(0)
        
        return effective_levels
    
    def get_sensitivity(self, nt_name: str = None) -> Union[float, Dict[str, float]]:
        """
        受容体感度を取得
        
        パラメータ:
        - nt_name: 神経伝達物質名（Noneの場合はすべて）
        
        戻り値:
        - 受容体感度
        """
        if nt_name:
            if nt_name in self.sensitivity:
                return self.sensitivity[nt_name]
            else:
                return None
        else:
            return self.sensitivity.copy()
    
    def get_all_sensitivities(self) -> Dict[str, float]:
        """すべての受容体感度を取得"""
        return self.sensitivity.copy()
    
    def get_effective_level(self, 
                            nt_name: str, 
                            level: float) -> float:
        """
        実効的なレベルを計算（単一の神経伝達物質）
        
        パラメータ:
        - nt_name: 神経伝達物質名
        - level: 実際のレベル
        
        戻り値:
        - 実効的なレベル
        """
        if nt_name in self.sensitivity:
            return min(1.0, level * self.sensitivity[nt_name])
        else:
            return level
    
    def apply_medication_effect(self, 
                                medication_type: str, 
                                strength: float = 1.0):
        """
        薬物の受容体への効果を適用
        
        パラメータ:
        - medication_type: 薬物の種類
        - strength: 効果の強さ（0.0～1.0）
        """
        # 様々な薬物タイプとその受容体への効果
        medications = {
            "ssri": {  # 選択的セロトニン再取り込み阻害薬
                "serotonin": 0.7,  # セロトニン受容体の感度低下
            },
            "stimulant": {  # 中枢神経刺激薬（アンフェタミンなど）
                "dopamine": 0.6,   # ドーパミン受容体の感度低下
                "noradrenaline": 0.7  # ノルアドレナリン受容体の感度低下
            },
            "benzodiazepine": {  # ベンゾジアゼピン系抗不安薬
                "gaba": 1.5  # GABA受容体の感度増強
            },
            "opioid": {  # オピオイド鎮痛薬
                "endorphin": 0.5  # エンドルフィン受容体の感度低下
            },
            "anticholinergic": {  # 抗コリン薬
                "acetylcholine": 0.4  # アセチルコリン受容体の感度低下
            },
            "dopamine_antagonist": {  # 抗精神病薬
                "dopamine": 0.3  # ドーパミン受容体の強力な感度低下
            },
            "glutamate_modulator": {  # グルタミン酸調節薬
                "glutamate": 0.8  # グルタミン酸受容体の感度調整
            }
        }
        
        if medication_type in medications:
            effects = medications[medication_type]
            
            # 効果の適用
            for nt, effect_factor in effects.items():
                if nt in self.sensitivity:
                    # 調整係数（効果の強さに比例）
                    adjusted_factor = 1.0 + (effect_factor - 1.0) * strength
                    
                    # 受容体感度に適用
                    self.sensitivity[nt] *= adjusted_factor
                    
                    # 範囲内に制限
                    self.sensitivity[nt] = max(0.1, min(3.0, self.sensitivity[nt]))
            
            print(f"{medication_type}の受容体への効果を適用しました（強さ: {strength}）")
        else:
            print(f"未知の薬物タイプ: {medication_type}")
    
    def plot_sensitivity_history(self, 
                                 save_path: str = None,
                                 selected_nt: List[str] = None):
        """
        受容体感度の履歴をプロット
        
        パラメータ:
        - save_path: 保存先パス（指定があれば保存）
        - selected_nt: 表示する神経伝達物質のリスト（Noneの場合はすべて）
        """
        if not self.time_points:
            print("データがありません")
            return
        
        plt.figure(figsize=(12, 6))
        
        if selected_nt is None:
            selected_nt = self.nt_names
        
        for nt in selected_nt:
            if nt in self.sensitivity_history:
                plt.plot(self.time_points, self.sensitivity_history[nt], label=f"{nt} 受容体")
        
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('時間')
        plt.ylabel('受容体感度')
        plt.title('神経伝達物質受容体感度の時間変化')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 2.0)
        
        if save_path:
            plt.savefig(save_path)
            print(f"グラフを保存しました: {save_path}")
        
        plt.show()
    
    def reset(self, initial_sensitivity: Dict[str, float] = None):
        """
        受容体システムをリセット
        
        パラメータ:
        - initial_sensitivity: リセット後の初期感度
        """
        # デフォルト感度に戻す
        default_sensitivity = {nt: 1.0 for nt in self.nt_names}
        
        # 初期感度を上書き
        if initial_sensitivity:
            for nt, sens in initial_sensitivity.items():
                if nt in default_sensitivity:
                    default_sensitivity[nt] = max(0.1, min(3.0, sens))
        
        # 感度の更新
        self.sensitivity = default_sensitivity
        
        # 受容体健全性のリセット
        self.receptor_health = {nt: 1.0 for nt in self.nt_names}
        
        # 履歴のクリア
        self.sensitivity_history = {nt: [] for nt in self.nt_names}
        self.time_points = []
        self.current_time = 0.0
        
        print("受容体フィードバックシステムをリセットしました")

class ReceptorGatingSystem:
    """
    神経伝達物質受容体のゲーティング機構をモデル化するクラス
    
    イオンチャネル開閉とシグナル伝達カスケードによる高次の調節効果を模倣します。
    """
    
    def __init__(self, nt_names: List[str] = None):
        """
        受容体ゲーティングシステムを初期化
        
        パラメータ:
        - nt_names: 神経伝達物質の名前リスト
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
        
        # チャネル開状態（0: 閉鎖、1: 全開）
        self.channel_states = {nt: 0.0 for nt in self.nt_names}
        
        # 開閉のダイナミクスパラメータ
        # 開速度: 高いほど速く開く
        self.opening_rate = {
            "acetylcholine": 0.8,   # 速い（ニコチン性受容体）
            "dopamine": 0.3,        # 遅い（代謝調節型）
            "serotonin": 0.3,       # 遅い（代謝調節型）
            "noradrenaline": 0.6,   # 中程度
            "glutamate": 0.9,       # 非常に速い（AMPA/NMDA）
            "gaba": 0.7,            # 速い（GABA-A）
            "endorphin": 0.5,       # 中程度
            "oxytocin": 0.3         # 遅い（代謝調節型）
        }
        
        # 閉鎖速度: 高いほど速く閉じる
        self.closing_rate = {
            "acetylcholine": 0.6,   # 中速（脱分極後）
            "dopamine": 0.2,        # 遅い（長期効果）
            "serotonin": 0.2,       # 遅い（長期効果）
            "noradrenaline": 0.5,   # 中程度
            "glutamate": 0.7,       # 速い（速い不活性化）
            "gaba": 0.5,            # 中程度
            "endorphin": 0.3,       # 遅い
            "oxytocin": 0.2         # 非常に遅い（長期効果）
        }
        
        # シグナル伝達効率（下流効果の強さ）
        self.signaling_efficiency = {
            "acetylcholine": 0.9,   # 高い
            "dopamine": 0.7,        # 中程度
            "serotonin": 0.7,       # 中程度
            "noradrenaline": 0.8,   # 高い
            "glutamate": 0.9,       # 高い
            "gaba": 0.8,            # 高い
            "endorphin": 0.7,       # 中程度
            "oxytocin": 0.6         # 中程度
        }
        
        # 閾値（効果が始まるレベル）
        self.thresholds = {nt: 0.2 for nt in self.nt_names}
        
        # 現在の信号強度（下流効果）
        self.current_signal = {nt: 0.0 for nt in self.nt_names}
        
        # 履歴
        self.channel_history = {nt: [] for nt in self.nt_names}
        self.signal_history = {nt: [] for nt in self.nt_names}
        self.time_points = []
        self.current_time = 0.0
        
        # 最大履歴サイズ
        self.max_history = 1000
        
        print("受容体ゲーティングシステム初期化完了")
    
    def update(self, 
               nt_levels: Dict[str, float], 
               dt: float = 0.1) -> Dict[str, float]:
        """
        受容体ゲーティングと信号伝達を更新
        
        パラメータ:
        - nt_levels: 実効的な神経伝達物質レベル（感度調整後）
        - dt: 時間ステップ
        
        戻り値:
        - 信号強度（下流効果）
        """
        # 時間を更新
        self.current_time += dt
        
        # 各神経伝達物質のチャネル状態を更新
        for nt in self.nt_names:
            if nt in nt_levels:
                level = nt_levels[nt]
                current_state = self.channel_states[nt]
                
                # チャネル開閉の計算
                if level > self.thresholds[nt]:
                    # 閾値を超えたらチャネルが開く傾向
                    opening = self.opening_rate[nt] * (level - self.thresholds[nt]) * dt
                    # 現在の開状態に応じた閉鎖
                    closing = self.closing_rate[nt] * current_state * dt
                    # 正味の変化
                    net_change = opening - closing
                else:
                    # 閾値以下なら閉じる一方
                    net_change = -self.closing_rate[nt] * current_state * dt
                
                # チャネル状態の更新（0～1の範囲に制限）
                self.channel_states[nt] = max(0.0, min(1.0, current_state + net_change))
                
                # 信号強度の計算（チャネル状態 × シグナル伝達効率）
                self.current_signal[nt] = self.channel_states[nt] * self.signaling_efficiency[nt]
        
        # 履歴の更新
        self.time_points.append(self.current_time)
        for nt in self.nt_names:
            self.channel_history[nt].append(self.channel_states[nt])
            self.signal_history[nt].append(self.current_signal[nt])
        
        # 履歴サイズの制限
        if len(self.time_points) > self.max_history:
            self.time_points.pop(0)
            for nt in self.nt_names:
                if len(self.channel_history[nt]) > self.max_history:
                    self.channel_history[nt].pop(0)
                if len(self.signal_history[nt]) > self.max_history:
                    self.signal_history[nt].pop(0)
        
        return self.current_signal.copy()
    
    def get_channel_states(self) -> Dict[str, float]:
        """チャネル開状態を取得"""
        return self.channel_states.copy()
    
    def get_signaling_strength(self) -> Dict[str, float]:
        """信号強度を取得"""
        return self.current_signal.copy()
    
    def plot_channel_dynamics(self, 
                              save_path: str = None,
                              selected_nt: List[str] = None):
        """
        チャネル動態の履歴をプロット
        
        パラメータ:
        - save_path: 保存先パス（指定があれば保存）
        - selected_nt: 表示する神経伝達物質のリスト（Noneの場合はすべて）
        """
        if not self.time_points:
            print("データがありません")
            return
        
        plt.figure(figsize=(12, 8))
        
        # サブプロットの作成
        plt.subplot(2, 1, 1)
        
        if selected_nt is None:
            selected_nt = self.nt_names
        
        # チャネル状態のプロット
        for nt in selected_nt:
            plt.plot(self.time_points, self.channel_history[nt], label=f"{nt} チャネル")
        
        plt.xlabel('時間')
        plt.ylabel('チャネル開状態')
        plt.title('受容体チャネルの開閉ダイナミクス')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.1)
        
        # 信号強度のプロット
        plt.subplot(2, 1, 2)
        
        for nt in selected_nt:
            plt.plot(self.time_points, self.signal_history[nt], label=f"{nt} 信号")
        
        plt.xlabel('時間')
        plt.ylabel('信号強度')
        plt.title('受容体シグナル伝達強度')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"グラフを保存しました: {save_path}")
        
        plt.show()
    
    def reset(self):
        """受容体ゲーティングシステムをリセット"""
        # チャネル状態と信号のリセット
        self.channel_states = {nt: 0.0 for nt in self.nt_names}
        self.current_signal = {nt: 0.0 for nt in self.nt_names}
        
        # 履歴のクリア
        self.channel_history = {nt: [] for nt in self.nt_names}
        self.signal_history = {nt: [] for nt in self.nt_names}
        self.time_points = []
        self.current_time = 0.0
        
        print("受容体ゲーティングシステムをリセットしました") 