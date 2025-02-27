"""
Genesis ユーティリティモジュール

このモジュールには、Genesisライブラリで使用される様々なユーティリティ関数と
ヘルパークラスが含まれています。

主な機能:
- 数学的ユーティリティ（四元数、行列変換など）
- データ記録とロギング
- 可視化ツール
- パフォーマンス測定
- ファイル入出力

詳細なドキュメント: https://genesis-world.readthedocs.io/en/latest/modules/utils.html
"""

import numpy as np
import time
import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# ロギング設定
logger = logging.getLogger("genesis")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class Timer:
    """
    コード実行時間の計測用タイマー
    """
    def __init__(self, name: str = "Timer"):
        """
        タイマーを初期化
        
        パラメータ:
        - name: このタイマーの名前
        """
        self.name = name
        self.start_time = None
        self.lap_time = None
        self.elapsed_time = 0.0
        self.running = False
        self.laps = []
    
    def start(self) -> None:
        """タイマーを開始"""
        if self.running:
            logger.warning(f"タイマー '{self.name}' は既に開始されています")
            return
        
        self.start_time = time.time()
        self.lap_time = self.start_time
        self.running = True
        self.elapsed_time = 0.0
        self.laps = []
        
        logger.debug(f"タイマー '{self.name}' 開始")
    
    def lap(self, label: str = None) -> float:
        """
        ラップタイムを記録
        
        パラメータ:
        - label: このラップのラベル
        
        戻り値:
        - 前回のラップ（または開始）からの経過時間
        """
        if not self.running:
            logger.warning(f"タイマー '{self.name}' が開始されていません")
            return 0.0
        
        current_time = time.time()
        lap_duration = current_time - self.lap_time
        self.lap_time = current_time
        
        lap_info = {
            "label": label or f"Lap {len(self.laps) + 1}",
            "time": lap_duration,
            "total_time": current_time - self.start_time
        }
        self.laps.append(lap_info)
        
        logger.debug(f"タイマー '{self.name}' ラップ: {lap_info['label']} = {lap_duration:.6f}秒")
        return lap_duration
    
    def stop(self) -> float:
        """
        タイマーを停止
        
        戻り値:
        - 合計経過時間
        """
        if not self.running:
            logger.warning(f"タイマー '{self.name}' は既に停止しています")
            return self.elapsed_time
        
        stop_time = time.time()
        self.elapsed_time = stop_time - self.start_time
        self.running = False
        
        logger.debug(f"タイマー '{self.name}' 停止: 合計 {self.elapsed_time:.6f}秒")
        return self.elapsed_time
    
    def reset(self) -> None:
        """タイマーをリセット"""
        self.start_time = None
        self.lap_time = None
        self.elapsed_time = 0.0
        self.running = False
        self.laps = []
        
        logger.debug(f"タイマー '{self.name}' リセット")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        タイマーの実行サマリーを取得
        
        戻り値:
        - タイマー情報を含む辞書
        """
        current_time = time.time()
        running_time = (current_time - self.start_time) if self.running else self.elapsed_time
        
        return {
            "name": self.name,
            "running": self.running,
            "elapsed_time": running_time,
            "lap_count": len(self.laps),
            "laps": self.laps
        }


class DataRecorder:
    """
    シミュレーションデータの記録と保存
    """
    def __init__(self, name: str = "simulation", directory: str = "./data"):
        """
        データレコーダーを初期化
        
        パラメータ:
        - name: レコーディングの名前
        - directory: データを保存するディレクトリ
        """
        self.name = name
        self.directory = directory
        self.records = []
        self.metadata = {
            "name": name,
            "start_time": time.time(),
            "version": "0.1.0"
        }
        
        # ディレクトリが存在しない場合は作成
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"ディレクトリを作成: {directory}")
    
    def record(self, data: Dict[str, Any], timestamp: float = None) -> None:
        """
        データポイントを記録
        
        パラメータ:
        - data: 記録するデータ（辞書形式）
        - timestamp: タイムスタンプ（指定しない場合は現在時刻）
        """
        if timestamp is None:
            timestamp = time.time()
        
        # データに時間情報を追加
        data_point = {
            "timestamp": timestamp,
            "data": data
        }
        
        self.records.append(data_point)
    
    def save(self, filename: str = None) -> str:
        """
        記録されたデータをJSONファイルとして保存
        
        パラメータ:
        - filename: 出力ファイル名（指定しない場合は自動生成）
        
        戻り値:
        - 保存されたファイルのパス
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}.json"
        
        file_path = os.path.join(self.directory, filename)
        
        # メタデータを更新
        self.metadata["end_time"] = time.time()
        self.metadata["duration"] = self.metadata["end_time"] - self.metadata["start_time"]
        self.metadata["record_count"] = len(self.records)
        
        # データを整形
        output_data = {
            "metadata": self.metadata,
            "records": self.records
        }
        
        # JSONとして保存
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"データを保存: {file_path} ({len(self.records)}レコード)")
        return file_path
    
    def clear(self) -> None:
        """記録を消去"""
        self.records = []
        self.metadata["start_time"] = time.time()
        logger.debug(f"レコーダー '{self.name}' のデータをクリア")


# 数学ユーティリティ
def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    2つの四元数の乗算
    
    パラメータ:
    - q1, q2: 四元数 [x, y, z, w]
    
    戻り値:
    - 乗算結果の四元数
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """
    四元数から回転行列への変換
    
    パラメータ:
    - q: 四元数 [x, y, z, w]
    
    戻り値:
    - 3x3回転行列
    """
    x, y, z, w = q
    
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def matrix_to_quaternion(m: np.ndarray) -> np.ndarray:
    """
    回転行列から四元数への変換
    
    パラメータ:
    - m: 3x3回転行列
    
    戻り値:
    - 四元数 [x, y, z, w]
    """
    tr = np.trace(m)
    
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    
    return np.array([x, y, z, w])

# ファイル操作ユーティリティ
def save_json(data: Dict[str, Any], filename: str) -> bool:
    """
    データをJSONファイルとして保存
    
    パラメータ:
    - data: 保存するデータ
    - filename: 出力ファイル名
    
    戻り値:
    - 成功した場合はTrue
    """
    try:
        # ディレクトリが存在しない場合は作成
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # NumPy配列をリストに変換
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(x) for x in obj]
            return obj
        
        processed_data = convert_numpy(data)
        
        # JSONとして保存
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"JSONを保存: {filename}")
        return True
    except Exception as e:
        logger.error(f"JSONの保存に失敗: {filename} - {e}")
        return False

def load_json(filename: str) -> Optional[Dict[str, Any]]:
    """
    JSONファイルからデータを読み込む
    
    パラメータ:
    - filename: 入力ファイル名
    
    戻り値:
    - 読み込まれたデータ、失敗した場合はNone
    """
    try:
        if not os.path.exists(filename):
            logger.error(f"ファイルが存在しません: {filename}")
            return None
        
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"JSONを読み込み: {filename}")
        return data
    except Exception as e:
        logger.error(f"JSONの読み込みに失敗: {filename} - {e}")
        return None

# パフォーマンスユーティリティ
def estimate_performance(function, args=None, kwargs=None, iterations=100) -> Dict[str, float]:
    """
    関数のパフォーマンスを推定
    
    パラメータ:
    - function: 測定する関数
    - args: 関数の位置引数
    - kwargs: 関数のキーワード引数
    - iterations: 実行回数
    
    戻り値:
    - パフォーマンス統計を含む辞書
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    
    # 準備実行（キャッシュウォームアップなど）
    for _ in range(5):
        function(*args, **kwargs)
    
    # 測定
    times = []
    for _ in range(iterations):
        start_time = time.time()
        function(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # 統計の計算
    times = np.array(times)
    
    return {
        "mean": float(np.mean(times)),
        "median": float(np.median(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "std": float(np.std(times)),
        "iterations": iterations
    } 