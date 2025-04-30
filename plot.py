# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import sys

# --- 設定 ---
# ★★★ 前回出力したCSVファイルへのパスを指定してください ★★★
CSV_FILE_PATH = 'tracking_results_opencv_tracker_nonetest.csv' 
# または 'tracking_results_yolo.csv', 'tracking_results_opticalflow.csv' など

# --- メイン処理 ---
try:
    # CSVファイルをPandas DataFrameとして読み込む
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"CSVファイル '{CSV_FILE_PATH}' を読み込みました。")
    # データの内容を少し表示 (確認用)
    print("\nデータの最初の5行:")
    print(df.head())
    print(f"\nデータ件数: {len(df)} フレーム")

except FileNotFoundError:
    print(f"エラー: CSVファイル '{CSV_FILE_PATH}' が見つかりません。パスを確認してください。")
    sys.exit()
except Exception as e:
    print(f"エラー: CSVファイルの読み込み中に問題が発生しました: {e}")
    sys.exit()

# --- グラフ描画 ---

# グラフ描画の準備
plt.figure(figsize=(12, 6)) # グラフのサイズを指定 (幅12インチ, 高さ6インチ)

# X軸に 'frame' (フレーム番号)、Y軸に 'distance_m' (計算された距離) をプロット
# 'distance_m' が None や数値でない場合の処理を追加
try:
    # 数値に変換できない値をNaN (Not a Number) にする
    df['distance_m_numeric'] = pd.to_numeric(df['distance_m'], errors='coerce')

    # NaNでない有効なデータだけをプロット
    valid_data = df.dropna(subset=['distance_m_numeric'])

    if not valid_data.empty:
        plt.plot(valid_data['frame'], valid_data['distance_m_numeric'], marker='.', linestyle='-', label='Distance (m)')

        # グラフの装飾
        plt.title('Runner Distance Over Time') # グラフタイトル
        plt.xlabel('Frame Number')            # X軸ラベル
        plt.ylabel('Distance from 0m (meters)') # Y軸ラベル
        plt.grid(True)                       # グリッド線を表示
        plt.legend()                         # 凡例を表示

        # Y軸の範囲を調整 (例: 0m から 55m など、必要に応じて)
        # plt.ylim(0, 55)

        # グラフを表示
        print("\nグラフを表示します...")
        plt.show()
    else:
        print("エラー: プロットできる有効な距離データがCSVファイルにありませんでした。")
        print("CSVファイルの内容を確認してください ('distance_m' 列)。")


except KeyError:
    print(f"エラー: CSVファイルに 'frame' または 'distance_m' という列が見つかりません。")
    print(f"CSVファイルの列名を確認してください。利用可能な列: {df.columns.tolist()}")
except Exception as e:
    print(f"エラー: グラフ描画中に予期せぬ問題が発生しました: {e}")