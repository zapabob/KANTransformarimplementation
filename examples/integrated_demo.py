#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
統合バイオミメティックシステムデモ

このスクリプトは統合されたバイオミメティックシステムの機能を実演します。
神経伝達物質、受容体フィードバック、機械学習が連携するさまざまなデモを含みます。
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple, Any
import threading

# モジュールのインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genesis.integrated_system import IntegratedBiomimeticSystem

def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description='統合バイオミメティックシステムデモ')
    parser.add_argument('--duration', type=int, default=60,
                      help='シミュレーション時間（秒）')
    parser.add_argument('--sensor_dim', type=int, default=10,
                      help='センサー入力の次元数')
    parser.add_argument('--motor_dim', type=int, default=10,
                      help='運動出力の次元数')
    parser.add_argument('--visualize', action='store_true',
                      help='リアルタイム可視化を有効にする')
    parser.add_argument('--save_state', action='store_true',
                      help='最終状態を保存する')
    parser.add_argument('--demo_type', type=str, default='all',
                      choices=['all', 'basic', 'learning', 'adaptation', 'medication'],
                      help='実行するデモタイプ')
    
    return parser.parse_args()

def generate_sensor_input(t: float, sensor_dim: int = 10) -> np.ndarray:
    """
    時間依存のセンサー入力を生成
    
    パラメータ:
    - t: 時間（秒）
    - sensor_dim: センサー次元数
    
    戻り値:
    - sensor_data: センサー入力データ
    """
    # 基本的な時間依存パターン
    base_pattern = np.zeros(sensor_dim)
    
    # 異なる周波数の正弦波を重ね合わせ
    for i in range(sensor_dim):
        freq = 0.1 + 0.02 * i  # 各次元ごとに異なる周波数
        phase = i * (2 * np.pi / sensor_dim)  # 位相をずらす
        base_pattern[i] = 0.5 + 0.5 * np.sin(t * freq + phase)
    
    # 周期的なバースト
    if int(t) % 10 == 0:  # 10秒ごとにバースト
        burst_idx = np.random.choice(sensor_dim, size=sensor_dim//3, replace=False)
        base_pattern[burst_idx] = 1.0
    
    # ノイズを追加
    noise = np.random.normal(0, 0.05, sensor_dim)
    
    return np.clip(base_pattern + noise, 0, 1)

def generate_target_motor(t: float, motor_dim: int = 10) -> np.ndarray:
    """
    時間依存の目標運動出力を生成
    
    パラメータ:
    - t: 時間（秒）
    - motor_dim: 運動出力の次元数
    
    戻り値:
    - target_motor: 目標運動データ
    """
    # 基本的な時間依存パターン
    target = np.zeros(motor_dim)
    
    # 円運動パターン（2次元の場合）
    if motor_dim >= 2:
        target[0] = 0.8 * np.cos(t * 0.2)
        target[1] = 0.8 * np.sin(t * 0.2)
    
    # 残りの次元にはその他のパターンを生成
    for i in range(2, motor_dim):
        freq = 0.05 + 0.01 * i
        amp = 0.5 + 0.2 * np.sin(t * 0.02)
        target[i] = amp * np.sin(t * freq)
    
    # 20秒ごとにパターンを反転
    if int(t) % 20 < 10:
        target = -target
    
    return np.clip(target, -1, 1)

def run_basic_demo(system: IntegratedBiomimeticSystem, duration: float = 60.0):
    """
    基本的なシステム動作のデモ
    
    パラメータ:
    - system: 統合システム
    - duration: 実行時間（秒）
    """
    print("\n===== 基本動作デモ実行中 =====")
    print("センサー入力と目標運動出力に反応するシステムの基本動作をデモします。")
    
    start_time = time.time()
    current_time = 0.0
    
    try:
        while current_time < duration:
            # 現在時刻を更新
            current_time = time.time() - start_time
            
            # センサー入力と目標運動出力を生成
            sensor_data = generate_sensor_input(current_time, system.sensor_dim)
            target_motor = generate_target_motor(current_time, system.motor_dim)
            
            # システムに入力を設定
            system.set_sensor_input(sensor_data)
            system.set_target_motor(target_motor)
            
            # 1秒ごとに状態を表示
            if int(current_time) > int(current_time - 0.1):
                nt_levels = system.nt_system.levels
                print(f"\r時間: {current_time:.1f}秒 | "
                      f"ACh: {nt_levels['acetylcholine']:.2f} | "
                      f"DA: {nt_levels['dopamine']:.2f} | "
                      f"5HT: {nt_levels['serotonin']:.2f} | "
                      f"NA: {nt_levels['noradrenaline']:.2f}", end="")
            
            # 少し待機
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\nデモが中断されました")
    
    print("\n基本動作デモ完了")

def run_learning_demo(system: IntegratedBiomimeticSystem, duration: float = 60.0):
    """
    学習動作のデモ
    
    パラメータ:
    - system: 統合システム
    - duration: 実行時間（秒）
    """
    print("\n===== 学習動作デモ実行中 =====")
    print("目標運動パターンを学習して再現するデモです。")
    
    # 学習データを記録するための配列
    times = []
    losses = []
    performances = []
    
    start_time = time.time()
    current_time = 0.0
    
    # 学習モードを有効化
    system.neural_control.training = True
    
    try:
        while current_time < duration:
            # 現在時刻を更新
            current_time = time.time() - start_time
            
            # センサー入力と目標運動出力を生成
            sensor_data = generate_sensor_input(current_time, system.sensor_dim)
            
            # 学習させるパターン（単純な正弦波）
            target_pattern = np.zeros(system.motor_dim)
            for i in range(system.motor_dim):
                phase = i * (2 * np.pi / system.motor_dim)
                target_pattern[i] = np.sin(current_time * 0.1 + phase)
            
            # システムに入力を設定
            system.set_sensor_input(sensor_data)
            system.set_target_motor(target_pattern)
            
            # 学習進捗を記録（1秒ごと）
            if int(current_time) > int(current_time - 0.1):
                # 現在のパフォーマンスを計算
                error = np.mean(np.abs(system.current_motor - target_pattern))
                performance = 1.0 - error
                
                times.append(current_time)
                performances.append(performance)
                
                print(f"\r時間: {current_time:.1f}秒 | "
                      f"パフォーマンス: {performance:.2f} | "
                      f"運動誤差: {error:.2f}", end="")
            
            # 少し待機
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\nデモが中断されました")
    
    # 学習曲線のプロット
    plt.figure(figsize=(10, 6))
    plt.plot(times, performances, 'b-')
    plt.xlabel('時間 (秒)')
    plt.ylabel('パフォーマンス')
    plt.title('運動パターン学習の進捗')
    plt.grid(True)
    
    # プロットを保存
    plt.savefig('learning_performance.png')
    print("\n学習曲線を保存しました: learning_performance.png")
    
    # 学習モードを無効化
    system.neural_control.training = False
    
    print("\n学習動作デモ完了")

def run_adaptation_demo(system: IntegratedBiomimeticSystem, duration: float = 90.0):
    """
    受容体適応のデモ
    
    パラメータ:
    - system: 統合システム
    - duration: 実行時間（秒）
    """
    print("\n===== 受容体適応デモ実行中 =====")
    print("神経伝達物質レベルに対する受容体感度の適応をデモします。")
    print("アセチルコリンレベルを変化させながら、受容体感度の変化を観察します。")
    
    # データを記録するための配列
    times = []
    ach_levels = []
    ach_sensitivities = []
    effective_levels = []
    
    start_time = time.time()
    current_time = 0.0
    phase = 0  # 0: 通常, 1: 高ACh, 2: 通常に戻る
    
    try:
        while current_time < duration:
            # 現在時刻を更新
            current_time = time.time() - start_time
            
            # フェーズによってアセチルコリンレベルを変更
            if current_time < 30:  # 最初の30秒は通常レベル
                ach_target = 0.5
                phase = 0
            elif current_time < 60:  # 30-60秒は高レベル
                ach_target = 0.9
                phase = 1
            else:  # 60秒以降は通常レベルに戻る
                ach_target = 0.5
                phase = 2
            
            # センサー入力（ACh生成を促すパターン）
            sensor_data = generate_sensor_input(current_time, system.sensor_dim)
            
            # アセチルコリンレベルを目標値に近づける
            current_ach = system.nt_system.levels["acetylcholine"]
            ach_change = (ach_target - current_ach) * 0.1
            system.nt_system.levels["acetylcholine"] = current_ach + ach_change
            
            # センサー入力を設定
            system.set_sensor_input(sensor_data)
            
            # データを記録（1秒ごと）
            if int(current_time) > int(current_time - 0.1):
                ach_level = system.nt_system.levels["acetylcholine"]
                ach_sensitivity = system.receptor_system.sensitivity["acetylcholine"]
                
                # 実効レベルを計算
                effective_level = ach_level * ach_sensitivity
                
                times.append(current_time)
                ach_levels.append(ach_level)
                ach_sensitivities.append(ach_sensitivity)
                effective_levels.append(effective_level)
                
                phase_labels = ["通常", "高ACh", "回復中"]
                print(f"\r時間: {current_time:.1f}秒 | "
                      f"フェーズ: {phase_labels[phase]} | "
                      f"ACh: {ach_level:.2f} | "
                      f"感度: {ach_sensitivity:.2f} | "
                      f"実効: {effective_level:.2f}", end="")
            
            # 少し待機
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\nデモが中断されました")
    
    # アセチルコリンレベルと受容体感度のプロット
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(times, ach_levels, 'b-', label='アセチルコリンレベル')
    plt.plot(times, effective_levels, 'r-', label='実効レベル')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvspan(30, 60, alpha=0.2, color='yellow', label='高ACh期間')
    plt.xlabel('時間 (秒)')
    plt.ylabel('レベル')
    plt.title('アセチルコリンレベルと実効レベル')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(times, ach_sensitivities, 'g-', label='アセチルコリン受容体感度')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.axvspan(30, 60, alpha=0.2, color='yellow', label='高ACh期間')
    plt.xlabel('時間 (秒)')
    plt.ylabel('感度')
    plt.title('アセチルコリン受容体感度の変化')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # プロットを保存
    plt.savefig('receptor_adaptation.png')
    print("\n受容体適応グラフを保存しました: receptor_adaptation.png")
    
    print("\n受容体適応デモ完了")

def run_medication_demo(system: IntegratedBiomimeticSystem, duration: float = 120.0):
    """
    薬物効果のデモ
    
    パラメータ:
    - system: 統合システム
    - duration: 実行時間（秒）
    """
    print("\n===== 薬物効果デモ実行中 =====")
    print("さまざまな薬物の神経伝達物質レベルと受容体感度への影響をデモします。")
    
    # 薬物リスト
    medications = [
        {"name": "SSRI", "type": "ssri", "time": 30, "desc": "セロトニン再取り込み阻害薬"},
        {"name": "興奮薬", "type": "stimulant", "time": 60, "desc": "ドーパミン・ノルアドレナリン増加"},
        {"name": "抗不安薬", "type": "benzodiazepine", "time": 90, "desc": "GABA受容体作用薬"}
    ]
    
    # データを記録するための配列
    times = []
    nt_data = {nt: [] for nt in system.nt_names}
    sensitivity_data = {nt: [] for nt in system.nt_names}
    med_times = []  # 投薬タイミング
    med_names = []  # 投薬名
    
    start_time = time.time()
    current_time = 0.0
    
    try:
        while current_time < duration:
            # 現在時刻を更新
            current_time = time.time() - start_time
            
            # センサー入力
            sensor_data = generate_sensor_input(current_time, system.sensor_dim)
            system.set_sensor_input(sensor_data)
            
            # 規定時間に達したら薬物効果を適用
            for med in medications:
                if int(current_time) == med["time"]:
                    print(f"\n時間 {current_time}秒: {med['name']}（{med['desc']}）を投与")
                    system.nt_system.simulate_medication(med["type"])
                    system.receptor_system.apply_medication_effect(med["type"])
                    
                    med_times.append(current_time)
                    med_names.append(med["name"])
            
            # データを記録（1秒ごと）
            if int(current_time) > int(current_time - 0.1):
                times.append(current_time)
                
                for nt in system.nt_names:
                    nt_data[nt].append(system.nt_system.levels[nt])
                    sensitivity_data[nt].append(system.receptor_system.sensitivity[nt])
                
                # 現在の薬物効果を表示
                active_med = next((m for m in medications if m["time"] <= current_time < m["time"] + 30), None)
                med_info = f"薬物: {active_med['name'] if active_med else 'なし'}"
                
                # 重要な神経伝達物質の状態を表示
                print(f"\r時間: {current_time:.1f}秒 | {med_info} | "
                      f"5HT: {system.nt_system.levels['serotonin']:.2f} | "
                      f"DA: {system.nt_system.levels['dopamine']:.2f} | "
                      f"GABA: {system.nt_system.levels['gaba']:.2f} | "
                      f"NA: {system.nt_system.levels['noradrenaline']:.2f}", end="")
            
            # 少し待機
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\nデモが中断されました")
    
    # 神経伝達物質レベルと受容体感度のプロット
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    for nt in ["serotonin", "dopamine", "noradrenaline", "gaba"]:
        plt.plot(times, nt_data[nt], label=nt)
    
    # 投薬タイミングにマーカー追加
    for i, t in enumerate(med_times):
        plt.axvline(x=t, color='r', linestyle='--', alpha=0.5)
        plt.text(t+1, 0.9, med_names[i], color='r')
    
    plt.xlabel('時間 (秒)')
    plt.ylabel('神経伝達物質レベル')
    plt.title('薬物投与による神経伝達物質レベルの変化')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    for nt in ["serotonin", "dopamine", "noradrenaline", "gaba"]:
        plt.plot(times, sensitivity_data[nt], label=f"{nt} 受容体")
    
    # 投薬タイミングにマーカー追加
    for i, t in enumerate(med_times):
        plt.axvline(x=t, color='r', linestyle='--', alpha=0.5)
    
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('時間 (秒)')
    plt.ylabel('受容体感度')
    plt.title('薬物投与による受容体感度の変化')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # プロットを保存
    plt.savefig('medication_effects.png')
    print("\n薬物効果グラフを保存しました: medication_effects.png")
    
    print("\n薬物効果デモ完了")

def main():
    """メイン関数"""
    args = parse_args()
    
    print("\n========= 統合バイオミメティックシステムデモ =========\n")
    print(f"設定: 時間={args.duration}秒, センサー次元={args.sensor_dim}, 運動次元={args.motor_dim}")
    
    # 統合システムの初期化
    system = IntegratedBiomimeticSystem(
        sensor_dim=args.sensor_dim,
        motor_dim=args.motor_dim,
        visualization=args.visualize
    )
    
    # システムの開始
    system.start()
    
    try:
        # 選択されたデモを実行
        if args.demo_type == 'all' or args.demo_type == 'basic':
            run_basic_demo(system, duration=min(30.0, args.duration))
        
        if args.demo_type == 'all' or args.demo_type == 'learning':
            run_learning_demo(system, duration=min(60.0, args.duration))
        
        if args.demo_type == 'all' or args.demo_type == 'adaptation':
            run_adaptation_demo(system, duration=min(90.0, args.duration))
        
        if args.demo_type == 'all' or args.demo_type == 'medication':
            run_medication_demo(system, duration=min(120.0, args.duration))
        
        # 状態の保存（オプション）
        if args.save_state:
            system.save_state()
    
    except KeyboardInterrupt:
        print("\nデモが中断されました")
    
    finally:
        # システムの停止
        system.stop()
        print("\nシステムを停止しました")
    
    print("\n=========== デモ完了 ===========")

if __name__ == "__main__":
    main() 