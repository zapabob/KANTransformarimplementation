#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
受容体フィードバック機能デモスクリプト

このスクリプトは以下の機能をデモンストレーションします：
1. 複数の神経伝達物質とその相互作用
2. 受容体の脱感作と再感作のダイナミクス
3. 薬物効果のシミュレーション
4. リアルタイム可視化
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple

# Genesisモジュールのインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genesis.neurotransmitters import NeurotransmitterSystem
from genesis.receptors import ReceptorFeedbackSystem, ReceptorGatingSystem
from genesis.visualizer import NeuralVisualizer

def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description='受容体フィードバックデモ')
    parser.add_argument('--duration', type=int, default=60, 
                        help='シミュレーション時間（秒）')
    parser.add_argument('--dt', type=float, default=0.1, 
                        help='時間ステップ（秒）')
    parser.add_argument('--visualize', action='store_true', 
                        help='リアルタイム可視化を有効にする')
    parser.add_argument('--save_plot', action='store_true', 
                        help='結果のプロットを保存する')
    parser.add_argument('--medication', type=str, default=None, 
                        choices=['ssri', 'stimulant', 'benzodiazepine', 'opioid', 'anticholinergic', 'dopamine_antagonist'],
                        help='シミュレーションする薬物タイプ')
    parser.add_argument('--medication_time', type=float, default=20.0, 
                        help='薬物を投与する時間（秒）')
    parser.add_argument('--selected_nts', type=str, nargs='+',
                        default=['acetylcholine', 'dopamine', 'serotonin', 'noradrenaline'],
                        help='表示する神経伝達物質（スペース区切り）')
    
    return parser.parse_args()

def run_neurotransmitter_simulation(args):
    """
    神経伝達物質シミュレーションを実行
    
    パラメータ:
    - args: コマンドライン引数
    """
    print("神経伝達物質シミュレーションを開始します...")
    
    # 神経伝達物質システムの初期化
    nt_system = NeurotransmitterSystem()
    
    # 受容体フィードバックシステムの初期化
    receptor_system = ReceptorFeedbackSystem()
    
    # 受容体ゲーティングシステムの初期化
    gating_system = ReceptorGatingSystem()
    
    # 可視化システムの初期化（指定があれば）
    visualizer = None
    if args.visualize:
        visualizer = NeuralVisualizer(update_interval=100)
        
        # 神経伝達物質プロットを設定
        visualizer.setup_neurotransmitter_plot(
            lambda: nt_system.levels
        )
        
        # 受容体感度プロットを設定
        visualizer.setup_receptor_sensitivity_plot(
            lambda: receptor_system.get_all_sensitivities()
        )
        
        # ダッシュボードの作成
        visualizer.create_dashboard()
        visualizer.start_visualization()
    
    # シミュレーションデータを保持するリスト
    time_points = []
    nt_levels_history = {nt: [] for nt in nt_system.levels}
    effective_levels_history = {nt: [] for nt in nt_system.levels}
    sensitivity_history = {nt: [] for nt in receptor_system.nt_names}
    
    # シミュレーションパラメータ
    dt = args.dt
    duration = args.duration
    current_time = 0.0
    
    # 外部刺激のパターン（時間依存）
    def get_external_inputs(t):
        # 基本的な周期的パターン
        inputs = {}
        
        # アセチルコリン: 5秒ごとにピーク
        inputs["acetylcholine"] = 0.3 + 0.6 * np.sin(t * 0.4) ** 2
        
        # ドーパミン: 15秒ごとにピーク
        inputs["dopamine"] = 0.2 + 0.7 * np.sin(t * 0.13) ** 2
        
        # セロトニン: 徐々に上昇して下降
        inputs["serotonin"] = 0.3 + 0.5 * np.sin(t * 0.05)
        
        # ノルアドレナリン: ランダムな急上昇
        if np.random.random() < 0.05:  # 5%の確率で急上昇
            inputs["noradrenaline"] = 0.8
        else:
            inputs["noradrenaline"] = 0.2 + 0.3 * np.sin(t * 0.2)
        
        # グルタミン酸: 周期的なバースト
        inputs["glutamate"] = 0.2 + 0.7 * (np.sin(t * 0.3) > 0.7)
        
        # GABA: グルタミン酸の逆相で変動
        inputs["gaba"] = 0.6 - 0.4 * (np.sin(t * 0.3) > 0.0)
        
        return inputs
    
    print(f"シミュレーション開始: 時間={duration}秒, ステップ={dt}秒")
    
    # 薬物効果の適用時間
    medication_applied = False
    
    # メインループ
    try:
        while current_time < duration:
            # 1. 外部入力の取得
            external_inputs = get_external_inputs(current_time)
            
            # 2. 神経伝達物質レベルの更新
            nt_levels = nt_system.update(dt, external_inputs)
            
            # 3. 薬物効果の適用（指定時間に達したとき）
            if not medication_applied and args.medication and current_time >= args.medication_time:
                print(f"薬物効果の適用: {args.medication} (時間: {current_time:.1f}秒)")
                nt_system.simulate_medication(args.medication)
                receptor_system.apply_medication_effect(args.medication)
                medication_applied = True
            
            # 4. 受容体フィードバックの更新
            effective_levels = receptor_system.update(nt_levels, dt)
            
            # 5. 受容体ゲーティングの更新
            signaling = gating_system.update(effective_levels, dt)
            
            # データの記録
            time_points.append(current_time)
            for nt in nt_system.levels:
                nt_levels_history[nt].append(nt_levels[nt])
                if nt in effective_levels:
                    effective_levels_history[nt].append(effective_levels[nt])
                else:
                    effective_levels_history[nt].append(0.0)
            
            for nt in receptor_system.nt_names:
                sensitivity_history[nt].append(receptor_system.sensitivity[nt])
            
            # 時間を進める
            current_time += dt
            
            # リアルタイムシミュレーションのためのスリープ（可視化中は特に重要）
            if args.visualize:
                time.sleep(dt * 0.1)  # 実時間より速く実行
    
    except KeyboardInterrupt:
        print("\nシミュレーションが中断されました")
    
    finally:
        # 可視化システムの停止
        if visualizer:
            visualizer.stop_visualization()
    
    print(f"シミュレーション完了: {len(time_points)}ステップ")
    
    # 結果のプロット
    plot_simulation_results(
        time_points, nt_levels_history, effective_levels_history, 
        sensitivity_history, args.selected_nts, args.save_plot,
        medication=args.medication, medication_time=args.medication_time
    )
    
    return time_points, nt_levels_history, effective_levels_history, sensitivity_history

def plot_simulation_results(time_points, nt_levels, effective_levels, sensitivities, 
                           selected_nts, save_plot=False, medication=None, medication_time=None):
    """
    シミュレーション結果をプロット
    
    パラメータ:
    - time_points: 時間データ
    - nt_levels: 神経伝達物質レベル履歴
    - effective_levels: 実効レベル履歴
    - sensitivities: 受容体感度履歴
    - selected_nts: 表示する神経伝達物質
    - save_plot: プロットを保存するかどうか
    - medication: 適用した薬物タイプ
    - medication_time: 薬物を適用した時間
    """
    plt.figure(figsize=(15, 12))
    
    # 1. 神経伝達物質レベル
    plt.subplot(3, 1, 1)
    for nt in selected_nts:
        if nt in nt_levels:
            plt.plot(time_points, nt_levels[nt], label=f"{nt}")
    
    plt.xlabel('時間 (秒)')
    plt.ylabel('レベル')
    plt.title('神経伝達物質レベル')
    plt.grid(True)
    plt.legend()
    
    # 薬物効果のマーク
    if medication and medication_time:
        plt.axvline(x=medication_time, color='r', linestyle='--', alpha=0.5)
        plt.text(medication_time + 0.5, 0.9, f"{medication}投与", color='r')
    
    # 2. 実効レベル（感度調整後）
    plt.subplot(3, 1, 2)
    for nt in selected_nts:
        if nt in effective_levels:
            plt.plot(time_points, effective_levels[nt], label=f"{nt} (実効)")
    
    plt.xlabel('時間 (秒)')
    plt.ylabel('実効レベル')
    plt.title('実効的な神経伝達物質レベル（受容体感度調整後）')
    plt.grid(True)
    plt.legend()
    
    # 薬物効果のマーク
    if medication and medication_time:
        plt.axvline(x=medication_time, color='r', linestyle='--', alpha=0.5)
    
    # 3. 受容体感度
    plt.subplot(3, 1, 3)
    for nt in selected_nts:
        if nt in sensitivities:
            plt.plot(time_points, sensitivities[nt], label=f"{nt} 受容体")
    
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('時間 (秒)')
    plt.ylabel('受容体感度')
    plt.title('神経伝達物質受容体の感度変化')
    plt.grid(True)
    plt.legend()
    
    # 薬物効果のマーク
    if medication and medication_time:
        plt.axvline(x=medication_time, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # プロットの保存
    if save_plot:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        med_str = f"_{medication}" if medication else ""
        filename = f"receptor_simulation_{timestamp}{med_str}.png"
        plt.savefig(filename)
        print(f"プロットを保存しました: {filename}")
    
    plt.show()

def demonstrate_desensitization():
    """
    受容体脱感作の特性をデモンストレーション
    
    高レベルの持続刺激による脱感作と回復をシミュレーション
    """
    print("受容体脱感作デモを開始します...")
    
    # 受容体フィードバックシステムの初期化
    receptor_system = ReceptorFeedbackSystem()
    
    # シミュレーションデータを保持するリスト
    time_points = []
    ach_levels = []
    effective_levels = []
    sensitivities = []
    
    # シミュレーションパラメータ
    dt = 0.1
    total_time = 120.0  # 2分間
    current_time = 0.0
    
    # ステージ：
    # 1. 通常レベル（0〜30秒）: アセチルコリン = 0.5
    # 2. 高レベル（30〜60秒）: アセチルコリン = 0.9
    # 3. 通常レベル（60〜90秒）: アセチルコリン = 0.5
    # 4. 低レベル（90〜120秒）: アセチルコリン = 0.2
    
    print("アセチルコリン受容体脱感作テスト：")
    print("1. 通常レベル（0〜30秒）")
    print("2. 高レベル（30〜60秒）- 脱感作が発生")
    print("3. 通常レベル（60〜90秒）- 部分的回復")
    print("4. 低レベル（90〜120秒）- 完全回復")
    
    while current_time < total_time:
        # 時間に応じたアセチルコリンレベル
        if 30 <= current_time < 60:
            ach_level = 0.9  # 高レベル
        elif 90 <= current_time < 120:
            ach_level = 0.2  # 低レベル
        else:
            ach_level = 0.5  # 通常レベル
        
        # 神経伝達物質レベルの設定
        nt_levels = {
            "acetylcholine": ach_level,
            "dopamine": 0.5,
            "serotonin": 0.5,
            "noradrenaline": 0.5
        }
        
        # 受容体フィードバックの更新
        effective_levels_dict = receptor_system.update(nt_levels, dt)
        
        # データの記録
        time_points.append(current_time)
        ach_levels.append(ach_level)
        effective_levels.append(effective_levels_dict["acetylcholine"])
        sensitivities.append(receptor_system.sensitivity["acetylcholine"])
        
        # 時間を進める
        current_time += dt
    
    # 結果のプロット
    plt.figure(figsize=(12, 8))
    
    # アセチルコリンレベル、実効レベル、受容体感度をプロット
    plt.plot(time_points, ach_levels, 'b-', label='アセチルコリンレベル')
    plt.plot(time_points, effective_levels, 'r-', label='実効レベル（レベル×感度）')
    plt.plot(time_points, sensitivities, 'g--', label='受容体感度')
    
    # 各フェーズを表示
    plt.axvspan(0, 30, alpha=0.1, color='gray', label='通常レベル')
    plt.axvspan(30, 60, alpha=0.2, color='red', label='高レベル（脱感作）')
    plt.axvspan(60, 90, alpha=0.1, color='blue', label='通常レベル（回復）')
    plt.axvspan(90, 120, alpha=0.2, color='green', label='低レベル（再感作）')
    
    plt.grid(True)
    plt.xlabel('時間 (秒)')
    plt.ylabel('レベル / 感度')
    plt.title('アセチルコリン受容体の脱感作と再感作ダイナミクス')
    plt.legend()
    
    # 基準線（感度=1.0）を追加
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # プロットの保存
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"desensitization_demo_{timestamp}.png"
    plt.savefig(filename)
    print(f"脱感作デモプロットを保存しました: {filename}")
    
    plt.show()
    
    print("\n脱感作特性の要約:")
    print(f"1. 初期感度: {sensitivities[0]:.2f}")
    print(f"2. 高刺激後の最低感度（60秒）: {sensitivities[int(60/dt)]:.2f}")
    print(f"3. 通常レベルでの回復後の感度（90秒）: {sensitivities[int(90/dt)]:.2f}")
    print(f"4. 低刺激後の最終感度（120秒）: {sensitivities[-1]:.2f}")
    
    return time_points, ach_levels, effective_levels, sensitivities

def demonstrate_medication_effects():
    """
    薬物効果のデモンストレーション
    
    異なる薬物の受容体感度への影響をシミュレーション
    """
    print("薬物効果デモを開始します...")
    
    # テストする薬物リスト
    medications = [
        "ssri",
        "stimulant",
        "benzodiazepine",
        "opioid",
        "anticholinergic"
    ]
    
    plt.figure(figsize=(15, 12))
    
    for i, medication in enumerate(medications):
        # 受容体フィードバックシステムの初期化
        receptor_system = ReceptorFeedbackSystem()
        
        # 影響を受ける主な神経伝達物質を特定
        affected_nts = []
        if medication == "ssri":
            affected_nts = ["serotonin"]
        elif medication == "stimulant":
            affected_nts = ["dopamine", "noradrenaline"]
        elif medication == "benzodiazepine":
            affected_nts = ["gaba"]
        elif medication == "opioid":
            affected_nts = ["endorphin"]
        elif medication == "anticholinergic":
            affected_nts = ["acetylcholine"]
        
        # シミュレーションデータを保持するリスト
        time_points = []
        sensitivity = {nt: [] for nt in affected_nts}
        
        # シミュレーションパラメータ
        dt = 0.1
        total_time = 30.0  # 30秒間
        current_time = 0.0
        medication_time = 10.0  # 10秒で薬物適用
        
        # メインループ
        while current_time < total_time:
            # 基本的な神経伝達物質レベル
            nt_levels = {nt: 0.5 for nt in receptor_system.nt_names}
            
            # 薬物効果の適用
            if current_time >= medication_time and current_time <= medication_time + dt:
                receptor_system.apply_medication_effect(medication)
            
            # 受容体フィードバックの更新
            receptor_system.update(nt_levels, dt)
            
            # データの記録
            time_points.append(current_time)
            for nt in affected_nts:
                sensitivity[nt].append(receptor_system.sensitivity[nt])
            
            # 時間を進める
            current_time += dt
        
        # サブプロットの作成
        plt.subplot(len(medications), 1, i+1)
        
        for nt in affected_nts:
            plt.plot(time_points, sensitivity[nt], label=f"{nt} 受容体")
        
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=medication_time, color='r', linestyle='--', alpha=0.5)
        plt.text(medication_time + 0.5, 1.1, f"{medication}投与", color='r')
        
        plt.xlabel('時間 (秒)')
        plt.ylabel('受容体感度')
        plt.title(f'{medication}の受容体感度への影響')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    
    # プロットの保存
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"medication_effects_demo_{timestamp}.png"
    plt.savefig(filename)
    print(f"薬物効果デモプロットを保存しました: {filename}")
    
    plt.show()
    
    print("\n薬物効果の要約:")
    for medication in medications:
        print(f"{medication}: 関連する受容体の感度を変化させます")

if __name__ == "__main__":
    args = parse_args()
    
    print("\n===== 受容体フィードバック機能デモ =====\n")
    
    # デモの実行
    if args.duration > 0:
        print("\n------- 神経伝達物質シミュレーション -------")
        run_neurotransmitter_simulation(args)
    
    print("\n------- 受容体脱感作デモ -------")
    demonstrate_desensitization()
    
    print("\n------- 薬物効果デモ -------")
    demonstrate_medication_effects()
    
    print("\nすべてのデモが完了しました。") 