"""
Genesis リアルタイム可視化モジュール

このモジュールは神経伝達物質レベル、運動出力、神経活動などをリアルタイムで
視覚化するための機能を提供します。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple, Callable, Optional, Union, Any
import threading
import time
import queue

class NeuralVisualizer:
    """
    神経活動やニューロモジュレーションをリアルタイムで可視化するクラス
    
    機能:
    - 神経伝達物質レベルのリアルタイムグラフ
    - 運動出力の視覚化
    - 神経活動のヒートマップ表示
    - 受容体感度の変化グラフ
    - インタラクティブな入力制御
    """
    
    def __init__(self, update_interval: int = 100):
        """
        可視化システムを初期化
        
        パラメータ:
        - update_interval: グラフ更新間隔(ミリ秒)
        """
        # 更新間隔
        self.update_interval = update_interval
        
        # プロットのデータソース（関数）
        self.data_sources = {}
        
        # 描画用のフィギュアとプロット
        self.fig = None
        self.axes = {}
        self.plots = {}
        
        # 時間履歴データ
        self.time_data = []
        self.max_time_points = 200
        
        # 実行状態
        self.running = False
        self.animation = None
        
        # アニメーション更新時の保護用ロック
        self.update_lock = threading.Lock()
        
        # 更新用スレッド
        self.update_thread = None
        
        # データキュー（スレッド間通信用）
        self.data_queue = queue.Queue()
        
        print("神経可視化システム初期化完了")
    
    def setup_neurotransmitter_plot(self, data_source: Callable[[], Dict[str, float]]):
        """
        神経伝達物質レベルのプロットを設定
        
        パラメータ:
        - data_source: 神経伝達物質レベルを取得する関数
        """
        self.data_sources['neurotransmitters'] = data_source
        
        # 初期データ
        initial_data = data_source()
        
        # 各神経伝達物質用の時系列データ
        self.nt_data = {name: [] for name in initial_data}
        
        print("神経伝達物質プロット設定完了")
    
    def setup_motor_output_plot(self, data_source: Callable[[], np.ndarray]):
        """
        運動出力のプロットを設定
        
        パラメータ:
        - data_source: 運動出力を取得する関数
        """
        self.data_sources['motor_output'] = data_source
        
        # 初期データ
        initial_data = data_source()
        
        # 運動出力の時系列データ
        self.motor_data = np.zeros((self.max_time_points, len(initial_data)))
        
        print("運動出力プロット設定完了")
    
    def setup_brain_activity_plot(self, data_source: Callable[[], np.ndarray]):
        """
        脳活動ヒートマップのプロットを設定
        
        パラメータ:
        - data_source: 脳活動データを取得する関数（2D配列）
        """
        self.data_sources['brain_activity'] = data_source
        
        # 初期データ
        self.brain_data = data_source()
        
        print("脳活動プロット設定完了")
    
    def setup_receptor_sensitivity_plot(self, data_source: Callable[[], Dict[str, float]]):
        """
        受容体感度プロットを設定
        
        パラメータ:
        - data_source: 受容体感度を取得する関数
        """
        self.data_sources['receptor_sensitivity'] = data_source
        
        # 初期データ
        initial_data = data_source()
        
        # 受容体感度の時系列データ
        self.receptor_data = {name: [] for name in initial_data}
        
        print("受容体感度プロット設定完了")
    
    def create_dashboard(self):
        """
        ダッシュボード（複数プロットを含む）を作成
        """
        # フィギュアの作成
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.suptitle('バイオミメティックニューラルシステム - リアルタイムモニタリング', fontsize=16)
        
        # グリッドレイアウトの作成
        gs = gridspec.GridSpec(3, 3, figure=self.fig)
        
        # 1. 神経伝達物質プロット（左上）
        if 'neurotransmitters' in self.data_sources:
            self.axes['nt'] = self.fig.add_subplot(gs[0, 0:2])
            self.axes['nt'].set_title('神経伝達物質レベル')
            self.axes['nt'].set_xlabel('時間')
            self.axes['nt'].set_ylabel('レベル')
            self.axes['nt'].set_ylim(0, 1.1)
            self.axes['nt'].grid(True)
            
            # 初期データ
            initial_data = self.data_sources['neurotransmitters']()
            
            # 各神経伝達物質の線を作成
            self.plots['nt'] = {}
            for i, (name, level) in enumerate(initial_data.items()):
                line, = self.axes['nt'].plot([], [], label=name)
                self.plots['nt'][name] = line
            
            self.axes['nt'].legend()
        
        # 2. 運動出力プロット（左下）
        if 'motor_output' in self.data_sources:
            self.axes['motor'] = self.fig.add_subplot(gs[1, 0:2])
            self.axes['motor'].set_title('運動出力')
            self.axes['motor'].set_xlabel('時間')
            self.axes['motor'].set_ylabel('出力値')
            self.axes['motor'].set_ylim(-1.1, 1.1)
            self.axes['motor'].grid(True)
            
            # 初期データ
            initial_data = self.data_sources['motor_output']()
            
            # 各運動出力の線を作成
            self.plots['motor'] = []
            for i in range(len(initial_data)):
                line, = self.axes['motor'].plot([], [], label=f'Motor {i+1}')
                self.plots['motor'].append(line)
            
            if len(initial_data) <= 10:  # 多すぎるとレジェンドが邪魔になる
                self.axes['motor'].legend()
        
        # 3. 脳活動ヒートマップ（右上）
        if 'brain_activity' in self.data_sources:
            self.axes['brain'] = self.fig.add_subplot(gs[0:2, 2])
            self.axes['brain'].set_title('神経活動マップ')
            
            # 初期データ
            self.brain_data = self.data_sources['brain_activity']()
            
            # ヒートマップ
            self.plots['brain'] = self.axes['brain'].imshow(
                self.brain_data, cmap='viridis', interpolation='nearest', vmin=0, vmax=1
            )
            self.fig.colorbar(self.plots['brain'], ax=self.axes['brain'], label='活動レベル')
        
        # 4. 受容体感度プロット（下）
        if 'receptor_sensitivity' in self.data_sources:
            self.axes['receptor'] = self.fig.add_subplot(gs[2, 0:3])
            self.axes['receptor'].set_title('受容体感度')
            self.axes['receptor'].set_xlabel('時間')
            self.axes['receptor'].set_ylabel('感度')
            self.axes['receptor'].set_ylim(0, 2.1)
            self.axes['receptor'].grid(True)
            
            # 初期データ
            initial_data = self.data_sources['receptor_sensitivity']()
            
            # 各受容体の線を作成
            self.plots['receptor'] = {}
            for i, (name, sensitivity) in enumerate(initial_data.items()):
                line, = self.axes['receptor'].plot([], [], label=f'{name} 受容体')
                self.plots['receptor'][name] = line
            
            self.axes['receptor'].legend()
        
        # レイアウトの最適化
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        print("ダッシュボード作成完了")
    
    def update_plots(self, frame):
        """
        プロットを更新（アニメーション用）
        """
        with self.update_lock:
            current_time = len(self.time_data)
            
            # 時間の更新
            if not self.time_data or (len(self.time_data) > 0 and self.time_data[-1] < current_time):
                self.time_data.append(current_time)
                if len(self.time_data) > self.max_time_points:
                    self.time_data.pop(0)
            
            # 1. 神経伝達物質レベルの更新
            if 'neurotransmitters' in self.data_sources and 'nt' in self.plots:
                try:
                    nt_levels = self.data_sources['neurotransmitters']()
                    
                    for name, level in nt_levels.items():
                        if name in self.nt_data:
                            self.nt_data[name].append(level)
                            if len(self.nt_data[name]) > self.max_time_points:
                                self.nt_data[name].pop(0)
                            
                            if name in self.plots['nt']:
                                self.plots['nt'][name].set_data(
                                    self.time_data[-len(self.nt_data[name]):], 
                                    self.nt_data[name]
                                )
                except Exception as e:
                    print(f"神経伝達物質データ更新エラー: {e}")
            
            # 2. 運動出力の更新
            if 'motor_output' in self.data_sources and 'motor' in self.plots:
                try:
                    motor_values = self.data_sources['motor_output']()
                    
                    # 新しいデータを追加
                    self.motor_data = np.roll(self.motor_data, -1, axis=0)
                    self.motor_data[-1, :] = motor_values
                    
                    # 各運動出力の線を更新
                    for i, line in enumerate(self.plots['motor']):
                        if i < len(motor_values):
                            line.set_data(
                                range(len(self.motor_data)), 
                                self.motor_data[:, i]
                            )
                            
                    # X軸の範囲を更新
                    if len(self.motor_data) > 0:
                        self.axes['motor'].set_xlim(0, len(self.motor_data))
                except Exception as e:
                    print(f"運動出力データ更新エラー: {e}")
            
            # 3. 脳活動マップの更新
            if 'brain_activity' in self.data_sources and 'brain' in self.plots:
                try:
                    self.brain_data = self.data_sources['brain_activity']()
                    self.plots['brain'].set_array(self.brain_data)
                except Exception as e:
                    print(f"脳活動データ更新エラー: {e}")
            
            # 4. 受容体感度の更新
            if 'receptor_sensitivity' in self.data_sources and 'receptor' in self.plots:
                try:
                    receptor_data = self.data_sources['receptor_sensitivity']()
                    
                    for name, sensitivity in receptor_data.items():
                        if name in self.receptor_data:
                            self.receptor_data[name].append(sensitivity)
                            if len(self.receptor_data[name]) > self.max_time_points:
                                self.receptor_data[name].pop(0)
                            
                            if name in self.plots['receptor']:
                                self.plots['receptor'][name].set_data(
                                    self.time_data[-len(self.receptor_data[name]):], 
                                    self.receptor_data[name]
                                )
                except Exception as e:
                    print(f"受容体感度データ更新エラー: {e}")
            
            # X軸の調整
            for ax_name in ['nt', 'receptor']:
                if ax_name in self.axes:
                    self.axes[ax_name].set_xlim(0, max(len(self.time_data), 1))
            
            return []
    
    def start_visualization(self):
        """
        可視化を開始
        """
        if self.fig is None:
            print("ダッシュボードが作成されていません。create_dashboard()を呼び出してください。")
            return
        
        if self.running:
            print("可視化は既に実行中です")
            return
        
        self.running = True
        
        # アニメーションの開始
        self.animation = FuncAnimation(
            self.fig, self.update_plots, interval=self.update_interval, 
            blit=True, cache_frame_data=False
        )
        
        # 非ブロッキングでプロットを表示
        plt.ion()  # インタラクティブモードを有効化
        plt.show(block=False)
        
        print("可視化を開始しました")
    
    def stop_visualization(self):
        """
        可視化を停止
        """
        if not self.running:
            return
        
        self.running = False
        
        # アニメーションを停止
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None
        
        # プロットを閉じる
        plt.close(self.fig)
        
        print("可視化を停止しました")
    
    def update_data_async(self):
        """
        非同期データ更新ループ
        """
        while self.running:
            try:
                # キューからデータを取得
                if not self.data_queue.empty():
                    data_type, data = self.data_queue.get(block=False)
                    
                    with self.update_lock:
                        # データ型に応じた処理
                        if data_type == 'nt' and 'nt' in self.plots:
                            for name, value in data.items():
                                if name in self.nt_data:
                                    self.nt_data[name].append(value)
                        elif data_type == 'motor' and 'motor' in self.plots:
                            self.motor_data = np.roll(self.motor_data, -1, axis=0)
                            self.motor_data[-1, :] = data
                        elif data_type == 'brain' and 'brain' in self.plots:
                            self.brain_data = data
                        elif data_type == 'receptor' and 'receptor' in self.plots:
                            for name, value in data.items():
                                if name in self.receptor_data:
                                    self.receptor_data[name].append(value)
                    
                    self.data_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"可視化データ更新エラー: {e}")
            
            time.sleep(self.update_interval / 2000)  # 更新間隔の半分
    
    def add_data(self, data_type: str, data: Any):
        """
        データキューに新しいデータを追加
        
        パラメータ:
        - data_type: データの種類 ('nt', 'motor', 'brain', 'receptor')
        - data: データの値
        """
        if self.running:
            self.data_queue.put((data_type, data))
    
    def __del__(self):
        """
        デストラクタ
        """
        self.stop_visualization() 