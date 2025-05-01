# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import sys
import platform
import pandas as pd
import numpy as np
import traceback
from PIL import Image, ImageTk
import ctypes # DPI Awareness用

# --- DPI Awareness 설정 (Windows) ---
try:
    if platform.system() == "Windows":
        ctypes.windll.shcore.SetProcessDpiAwareness(1) # または 2 を試す
        print("[INFO] DPI Awareness set for Windows.")
except AttributeError:
    print("[INFO] Could not set DPI Awareness (maybe not Windows or old OS/ctypes).")
except Exception as e:
    print(f"[WARN] Failed to set DPI Awareness: {e}")


# --- 設定 (変更可能) ---
MIN_ROI_WIDTH = 20
MIN_ROI_HEIGHT = 20
DEFAULT_OUTPUT_CSV = 'tracking_results.csv'

# --- グローバル変数 ---
video_path = ""
initial_frame = None        # 元の解像度のフレーム (numpy array)
# === リサイズ関連 ===
resized_frame = None      # キャンバス表示用にリサイズされたフレーム (numpy array)
display_image = None      # Tkinter表示用 (PhotoImage)
tk_image_ref = None       # PhotoImageの参照保持用
scale_factor = 1.0        # 元フレームと表示フレームのスケール比
offset_x = 0              # 表示フレームの描画オフセットX (中央揃え用)
offset_y = 0              # 表示フレームの描画オフセットY (中央揃え用)
# ==================
x_pix_0m = None           # ★ 元の解像度でのX座標として保存
x_pix_50m = None           # ★ 元の解像度でのX座標として保存
bbox = None             # ★ 元の解像度でのROI (x, y, w, h) として保存
roi_start_point = None    # キャンバス上のクリック開始点
selecting_roi = False
current_step = "LOAD_VIDEO"
results_data = []
tracker = None
tracker_name = ""
setup_complete = False # setup_complete フラグを追加

# --- トラッカー初期化関数 (変更なし) ---
def try_init_tracker(tracker_type_str, frame, roi):
    # (前回のコードと同じ)
    tracker_obj = None; success = False; actual_name = ""
    print(f"\n--- Trying to initialize {tracker_type_str} tracker ---")
    print(f"[DEBUG] Input frame shape: {frame.shape}, dtype: {frame.dtype}")
    print(f"[DEBUG] Input ROI value: {roi}, type: {type(roi)}")
    if roi is None or not isinstance(roi, tuple) or len(roi) != 4 or not all(isinstance(n, int) for n in roi):
        print(f"[DEBUG] Error: Invalid ROI {roi}."); return None, "", False
    creator_func = None
    try:
        if tracker_type_str == "CSRT":
            if hasattr(cv2, "TrackerCSRT_create"): creator_func = cv2.TrackerCSRT_create; actual_name = "CSRT"
            elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"): creator_func = cv2.legacy.TrackerCSRT_create; actual_name = "Legacy CSRT"
        elif tracker_type_str == "KCF":
             if hasattr(cv2, "TrackerKCF_create"): creator_func = cv2.TrackerKCF_create; actual_name = "KCF"
             elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"): creator_func = cv2.legacy.TrackerKCF_create; actual_name = "Legacy KCF"
        if creator_func:
            print(f"[DEBUG] Attempting to create tracker: {actual_name}")
            tracker_obj = creator_func()
            if tracker_obj is not None:
                print(f"[DEBUG] Tracker object created.")
                print(f"[DEBUG] Calling {actual_name}.init(frame, roi={roi})...")
                init_result = tracker_obj.init(frame, roi)
                print(f"[DEBUG] {actual_name}.init() returned: {init_result} (Type: {type(init_result)})")
                success = (init_result is not False) # Noneも成功扱い
                if not success: print(f"[DEBUG] Init returned False.")
                elif init_result is None: print(f"[DEBUG] WARNING: init() returned None. Assuming success...")
            else: print(f"[DEBUG] Error: Failed to create {actual_name} object."); return None, actual_name, False
        else: print(f"[DEBUG] Error: Creator for {tracker_type_str} not found."); return None, "", False
        if success:
            if init_result is not None: print(f"{actual_name} initialized successfully.")
            return tracker_obj, actual_name, True
        else: print(f"Failed to initialize {actual_name}."); return None, actual_name, False
    except Exception as e: print(f"[DEBUG] !!! Exception during {actual_name if actual_name else tracker_type_str} init !!!"); print(f"[DEBUG] Exception: {e}"); traceback.print_exc(); return None, actual_name, False


# --- GUIイベントハンドラ ---

def load_video():
    global video_path, initial_frame, current_step, x_pix_0m, x_pix_50m, bbox, setup_complete, results_data, tracker, tracker_name, roi_start_point, selecting_roi
    fpath = filedialog.askopenfilename(title="動画ファイルを選択", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
    if not fpath: return
    video_path = fpath
    filepath_var.set(f"読込中: {video_path}"); status_var.set("動画を読み込み中..."); root.update_idletasks()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): messagebox.showerror("エラー", f"動画ファイルを開けません:\n{video_path}"); filepath_var.set("動画未読込"); status_var.set("準備完了"); return
    ret, frame = cap.read(); cap.release()
    if not ret: messagebox.showerror("エラー", f"フレーム読込失敗:\n{video_path}"); filepath_var.set("動画未読込"); status_var.set("準備完了"); return
    initial_frame = frame
    filepath_var.set(f"読込完了: {video_path}")
    # リセット
    x_pix_0m = None; x_pix_50m = None; bbox = None; results_data = []; tracker = None; roi_start_point = None; selecting_roi = False; setup_complete = False
    current_step = "SELECT_0M"
    # キャンバスサイズを取得して最初の描画
    canvas.update_idletasks() # サイズ確定のため
    resize_and_show_frame(canvas.winfo_width(), canvas.winfo_height())
    status_var.set("ステップ 1/3: 画像上で「0m地点」をクリックしてください。")
    start_button.config(state=tk.DISABLED); save_button.config(state=tk.DISABLED)

def resize_and_show_frame(canvas_w, canvas_h):
    """フレームをキャンバスサイズに合わせてリサイズし、表示する"""
    global resized_frame, display_image, tk_image_ref, scale_factor, offset_x, offset_y

    if initial_frame is None or canvas_w <= 1 or canvas_h <= 1: # 初期化前やサイズ未確定時は何もしない
        return

    frame_h, frame_w = initial_frame.shape[:2]
    
    # アスペクト比を維持してリサイズ係数を計算
    scale_w = canvas_w / frame_w
    scale_h = canvas_h / frame_h
    scale_factor = min(scale_w, scale_h) # 小さい方に合わせる

    # 新しいサイズを計算
    new_w = int(frame_w * scale_factor)
    new_h = int(frame_h * scale_factor)

    # リサイズ実行 (高画質補間: INTER_AREA)
    resized_frame = cv2.resize(initial_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 中央揃えのためのオフセット計算
    offset_x = (canvas_w - new_w) // 2
    offset_y = (canvas_h - new_h) // 2

    # 表示用に変換
    img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    display_image = ImageTk.PhotoImage(img_pil)
    tk_image_ref = display_image # 重要: 参照保持

    # キャンバスに描画
    canvas.delete("all")
    canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=display_image, tags="frame")
    redraw_overlays() # 点やROIなどのオーバーレイを描画

def redraw_overlays():
    """現在の状態に基づいてオーバーレイ (点、ROI) を描画"""
    if resized_frame is None: return

    # 以前のオーバーレイを削除
    canvas.delete("overlays")

    # 0m地点 (元の座標をスケール変換して描画)
    if x_pix_0m is not None:
        # 元のフレームでのY座標が必要だが、簡単のため中心を使う
        orig_y = initial_frame.shape[0] // 2
        disp_x, disp_y = original_to_display_coords(x_pix_0m, orig_y)
        if disp_x is not None:
            canvas.create_oval(disp_x-3, disp_y-3, disp_x+3, disp_y+3, fill="lime", outline="lime", tags="overlays")
            canvas.create_text(disp_x + 10, disp_y, text="0m", fill="lime", anchor=tk.W, tags="overlays")

    # 50m地点
    if x_pix_50m is not None:
        orig_y = initial_frame.shape[0] // 2
        disp_x, disp_y = original_to_display_coords(x_pix_50m, orig_y)
        if disp_x is not None:
            canvas.create_oval(disp_x-3, disp_y-3, disp_x+3, disp_y+3, fill="red", outline="red", tags="overlays")
            canvas.create_text(disp_x + 10, disp_y, text="50m", fill="red", anchor=tk.W, tags="overlays")

    # 確定済みROI
    if bbox is not None:
        x1, y1, w, h = bbox # これは元の座標
        disp_x1, disp_y1 = original_to_display_coords(x1, y1)
        disp_x2, disp_y2 = original_to_display_coords(x1 + w, y1 + h)
        if disp_x1 is not None:
            canvas.create_rectangle(disp_x1, disp_y1, disp_x2, disp_y2, outline="blue", width=2, tags="overlays")

def display_to_original_coords(display_x, display_y):
    """表示座標 (キャンバス上) を元のフレーム座標に変換"""
    if scale_factor == 0: return None, None # まだ計算できない
    # オフセットとスケールを考慮して逆変換
    original_x = (display_x - offset_x) / scale_factor
    original_y = (display_y - offset_y) / scale_factor

    # 元画像の範囲内に収める (念のため)
    h, w = initial_frame.shape[:2]
    original_x = np.clip(original_x, 0, w - 1)
    original_y = np.clip(original_y, 0, h - 1)

    return int(original_x), int(original_y)

def original_to_display_coords(original_x, original_y):
    """元のフレーム座標を表示座標 (キャンバス上) に変換"""
    if scale_factor == 0: return None, None
    display_x = int(original_x * scale_factor + offset_x)
    display_y = int(original_y * scale_factor + offset_y)
    return display_x, display_y

def on_canvas_click(event):
    global current_step, x_pix_0m, x_pix_50m, roi_start_point, selecting_roi, bbox, setup_complete

    if initial_frame is None: return # 画像読込前は何もしない

    disp_x, disp_y = event.x, event.y # クリックされたキャンバス座標
    orig_x, orig_y = display_to_original_coords(disp_x, disp_y) # 元画像座標に変換
    print(f"[DEBUG] Canvas click at ({disp_x}, {disp_y}) -> Original ({orig_x}, {orig_y})")

    if orig_x is None: return # 変換失敗

    if current_step == "SELECT_0M":
        x_pix_0m = orig_x # ★ 元画像のX座標を保存
        status_var.set("ステップ 2/3: 画像上で「50m地点」をクリックしてください。")
        current_step = "SELECT_50M"
        redraw_overlays() # オーバーレイだけ再描画
    elif current_step == "SELECT_50M":
        x_pix_50m = orig_x # ★ 元画像のX座標を保存
        if x_pix_0m is not None:
            if abs(x_pix_0m - x_pix_50m) < 10: print("警告: 0mと50mが近すぎます！")
            elif x_pix_0m > x_pix_50m: print("警告: 0mが50mより右側です！")
        status_var.set("ステップ 3/3: 追跡対象をドラッグで選択してください。")
        current_step = "SELECT_ROI"
        redraw_overlays()
    elif current_step == "SELECT_ROI":
        selecting_roi = True
        roi_start_point = (disp_x, disp_y) # ★ ドラッグ開始点は表示座標
        bbox = None; setup_complete = False
        status_var.set("ROI選択中...")
        start_button.config(state=tk.DISABLED)
        redraw_overlays() # 以前のROIなどを消す

def on_canvas_drag(event):
    if not selecting_roi or roi_start_point is None: return
    disp_x, disp_y = event.x, event.y
    # ドラッグ中の矩形を表示座標で描画
    canvas.delete("feedback_roi") # 前の矩形を削除
    canvas.create_rectangle(roi_start_point[0], roi_start_point[1], disp_x, disp_y,
                            outline="lime", width=2, tags="feedback_roi")

def on_canvas_release(event):
    global selecting_roi, bbox, setup_complete
    if not selecting_roi or roi_start_point is None: return
    selecting_roi = False
    canvas.delete("feedback_roi") # ドラッグ中の矩形を削除

    # 終了点の表示座標
    disp_x1, disp_y1 = roi_start_point
    disp_x2, disp_y2 = event.x, event.y

    # 元のフレーム座標に変換
    orig_x1, orig_y1 = display_to_original_coords(disp_x1, disp_y1)
    orig_x2, orig_y2 = display_to_original_coords(disp_x2, disp_y2)

    if orig_x1 is None or orig_x2 is None:
        messagebox.showerror("エラー", "ROI座標の変換に失敗しました。")
        return

    # 元のフレーム上でのROI (x, y, w, h) を計算
    w = abs(orig_x1 - orig_x2)
    h = abs(orig_y1 - orig_y2)
    roi_x = min(orig_x1, orig_x2)
    roi_y = min(orig_y1, orig_y2)

    print(f"[DEBUG] ROI Release. Orig coords: ({roi_x}, {roi_y}, {w}, {h})")

    if w < MIN_ROI_WIDTH or h < MIN_ROI_HEIGHT:
        messagebox.showwarning("ROIエラー", f"選択領域が小さすぎます。\n最低{MIN_ROI_WIDTH}x{MIN_ROI_HEIGHT}ピクセル必要です。")
        bbox = None; setup_complete = False
        status_var.set("ROI選択失敗。再度ドラッグしてください。")
        start_button.config(state=tk.DISABLED)
    else:
        bbox = (int(roi_x), int(roi_y), int(w), int(h)) # ★ 元座標で保存
        setup_complete = True
        status_var.set(f"設定完了。ROI={bbox}。トラッキングを開始できます。")
        start_button.config(state=tk.NORMAL)
        
    redraw_overlays() # 確定ROIを描画

def on_canvas_configure(event):
    """キャンバスサイズ変更時のイベントハンドラ"""
    print(f"[DEBUG] Canvas configured to: {event.width} x {event.height}")
    resize_and_show_frame(event.width, event.height)

def start_tracking():
    global tracker, tracker_name, results_data, current_step
    if not setup_complete or initial_frame is None or bbox is None:
        messagebox.showerror("エラー", "開始前に動画読込、0m/50m/ROI設定を完了してください。")
        return
    status_var.set("トラッカー初期化中..."); load_button.config(state=tk.DISABLED); start_button.config(state=tk.DISABLED); save_button.config(state=tk.DISABLED); root.update_idletasks()
    # ★ 初期化には元の解像度のフレームを使う
    tracker_obj, name, success = try_init_tracker("CSRT", initial_frame, bbox)
    if not success:
        print("CSRT失敗、KCF試行..."); tracker_obj, name, success = try_init_tracker("KCF", initial_frame, bbox)
    if not success:
        messagebox.showerror("初期化エラー", "トラッカー初期化失敗。\nログ確認、ROI変更など試してください。"); status_var.set("初期化失敗"); load_button.config(state=tk.NORMAL); return
    tracker = tracker_obj; tracker_name = name; current_step = "TRACKING"
    status_var.set(f"トラッキング開始 ({tracker_name})..."); root.update_idletasks()
    # --- トラッキング実行 ---
    results_data = []
    cap = cv2.VideoCapture(video_path); cap.set(cv2.CAP_PROP_POS_FRAMES, 0); frame_number = -1; total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_var.set(0); progressbar['maximum'] = total_frames
    while True:
        frame_number += 1; ret, frame = cap.read()
        if not ret: break
        if frame_number % 10 == 0: progress_var.set(frame_number); status_var.set(f"処理中: {frame_number}/{total_frames}"); root.update_idletasks()
        success, new_bbox_float = tracker.update(frame)
        target_found = success; current_x_pix = None; distance_m = None; current_bbox_int = None
        if success:
            current_bbox_int = tuple(map(int, new_bbox_float))
            # ★ 距離計算には元のフレーム座標系でのX座標を使う
            original_bbox_x = current_bbox_int[0]; original_bbox_w = current_bbox_int[2]
            current_x_pix = original_bbox_x + original_bbox_w / 2 # 下辺中央ではなくボックス中心Xで計算してみる（より安定するかも）
            # current_x_pix = current_bbox_int[0] + current_bbox_int[2] / 2 # 元の下辺中央X
            if x_pix_0m is not None and x_pix_50m is not None and x_pix_50m != x_pix_0m:
                distance_m = 50.0 * (current_x_pix - x_pix_0m) / (x_pix_50m - x_pix_0m)
        results_data.append({
            'frame': frame_number, 'success': target_found, 'tracker': tracker_name,
            'pixel_x_center': current_x_pix if target_found else None, 'distance_m': distance_m if target_found else None, # 列名変更
            'bbox_x': current_bbox_int[0] if target_found else None, 'bbox_y': current_bbox_int[1] if target_found else None,
            'bbox_w': current_bbox_int[2] if target_found else None, 'bbox_h': current_bbox_int[3] if target_found else None,})
    cap.release(); progress_var.set(total_frames); status_var.set(f"完了。{len(results_data)}フレーム処理。")
    load_button.config(state=tk.NORMAL); start_button.config(state=tk.NORMAL);
    if results_data: save_button.config(state=tk.NORMAL)

def save_csv():
    if not results_data: messagebox.showwarning("保存エラー", "データがありません。"); return
    save_path = filedialog.asksaveasfilename(title="CSV保存", defaultextension=".csv", initialfile=DEFAULT_OUTPUT_CSV, filetypes=[("CSV", "*.csv"), ("All", "*.*")])
    if not save_path: return
    try: df = pd.DataFrame(results_data); df.to_csv(save_path, index=False, encoding='utf-8-sig'); messagebox.showinfo("完了", f"保存しました:\n{save_path}"); status_var.set(f"CSV保存完了: {save_path}")
    except Exception as e: messagebox.showerror("保存エラー", f"エラー:\n{e}"); status_var.set("CSV保存エラー")

# --- GUIセットアップ ---
root = tk.Tk(); root.title("動画ランナートラッカー"); root.geometry("900x700") # サイズ少し大きく

top_frame = tk.Frame(root); top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
load_button = tk.Button(top_frame, text="動画ファイルを開く", command=load_video); load_button.pack(side=tk.LEFT)
filepath_var = tk.StringVar(value="動画未読込"); filepath_label = tk.Label(top_frame, textvariable=filepath_var, anchor="w", relief=tk.SUNKEN); filepath_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

canvas_frame = tk.Frame(root, bg="dark gray"); canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5) # 背景色付けて範囲確認
canvas = tk.Canvas(canvas_frame, bg="black") # キャンバス背景
canvas.pack(fill=tk.BOTH, expand=True)
canvas.bind("<Configure>", on_canvas_configure) # ★★★ サイズ変更イベントをバインド ★★★
canvas.bind("<Button-1>", on_canvas_click); canvas.bind("<B1-Motion>", on_canvas_drag); canvas.bind("<ButtonRelease-1>", on_canvas_release)

bottom_frame = tk.Frame(root); bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
start_button = tk.Button(bottom_frame, text="トラッキング開始", command=start_tracking, state=tk.DISABLED); start_button.pack(side=tk.LEFT, padx=5)
save_button = tk.Button(bottom_frame, text="CSV保存", command=save_csv, state=tk.DISABLED); save_button.pack(side=tk.LEFT, padx=5)
status_var = tk.StringVar(value="準備完了"); status_label = tk.Label(bottom_frame, textvariable=status_var, anchor="w"); status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
progress_var = tk.DoubleVar(); progressbar = ttk.Progressbar(bottom_frame, variable=progress_var, orient="horizontal", length=200, mode="determinate"); progressbar.pack(side=tk.RIGHT, padx=5)

root.mainloop()