# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import sys
import platform
import pandas as pd
import numpy as np
import traceback
from PIL import Image, ImageTk
import ctypes
import threading
import queue
# ★ Matplotlib関連のインポート
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- DPI Awareness 설정 (Windows) ---
try:
    if platform.system() == "Windows": ctypes.windll.shcore.SetProcessDpiAwareness(1); print("[INFO] DPI Awareness set.")
except Exception as e: print(f"[WARN] Failed to set DPI Awareness: {e}")

# --- 設定 ---
MIN_ROI_WIDTH = 20; MIN_ROI_HEIGHT = 20
DEFAULT_OUTPUT_CSV = 'tracking_results.csv'
DEFAULT_FPS_IF_MISSING = 30.0
DEFAULT_MOVING_AVG_SEC = 0.5
PREVIEW_UPDATE_INTERVAL_MS = 30

# --- グローバル変数 ---
video_path = ""; video_fps = DEFAULT_FPS_IF_MISSING
initial_frame = None; resized_frame = None; display_image = None; tk_image_ref = None
scale_factor = 1.0; offset_x = 0; offset_y = 0
x_pix_0m = None; y_pix_0m = None; x_pix_50m = None; y_pix_50m = None
bbox = None; roi_start_point = None; selecting_roi = False
current_step = "LOAD_VIDEO"; setup_complete = False
results_data = []; tracker = None; tracker_name = ""
distance_offset = 0.0
mode_var = None; moving_avg_sec_var = None
tracking_thread = None; tracking_queue = queue.Queue(); stop_event = threading.Event()
graph_window = None
graph_df = None # グラフ用データ

# --- トラッカー初期化関数 ---
def try_init_tracker(tracker_type_str, frame, roi):
    tracker_obj = None; success = False; actual_name = ""
    print(f"\n--- Trying {tracker_type_str} tracker ---")
    print(f"[DEBUG] Frame: {frame.shape}, {frame.dtype}, ROI: {roi}, {type(roi)}")
    if roi is None or not isinstance(roi, tuple) or len(roi)!=4 or not all(isinstance(n, int) for n in roi): print(f"[DEBUG] Invalid ROI {roi}"); return None, "", False
    creator_func = None
    try:
        if tracker_type_str == "CSRT":
            if hasattr(cv2, "TrackerCSRT_create"): creator_func=cv2.TrackerCSRT_create; actual_name="CSRT"
            elif hasattr(cv2,"legacy") and hasattr(cv2.legacy,"TrackerCSRT_create"): creator_func=cv2.legacy.TrackerCSRT_create; actual_name="Legacy CSRT"
        elif tracker_type_str == "KCF":
             if hasattr(cv2, "TrackerKCF_create"): creator_func=cv2.TrackerKCF_create; actual_name="KCF"
             elif hasattr(cv2,"legacy") and hasattr(cv2.legacy,"TrackerKCF_create"): creator_func=cv2.legacy.TrackerKCF_create; actual_name="Legacy KCF"
        if creator_func:
            print(f"[DEBUG] Creating {actual_name}"); tracker_obj = creator_func()
            if tracker_obj:
                print(f"[DEBUG] Calling init..."); init_result = tracker_obj.init(frame, roi)
                print(f"[DEBUG] init returned: {init_result} ({type(init_result)})"); success = (init_result is not False)
                if not success: print(f"[DEBUG] Init returned False.")
                elif init_result is None: print(f"[DEBUG] WARNING: init returned None. Assuming success...")
            else: print(f"[DEBUG] Failed to create {actual_name}."); return None, actual_name, False
        else: print(f"[DEBUG] Creator for {tracker_type_str} not found."); return None, "", False
        if success:
            if init_result is not None: print(f"{actual_name} initialized."); return tracker_obj, actual_name, True
            else: return tracker_obj, actual_name, True # Noneでも成功扱い
        else: print(f"Failed to initialize {actual_name}."); return None, actual_name, False
    except Exception as e: print(f"[DEBUG] !!! Exception during {actual_name if actual_name else tracker_type_str} init !!!"); print(f"[DEBUG] {e}"); traceback.print_exc(); return None, actual_name, False

# --- GUIイベントハンドラ ---
def update_ui_for_mode(*args):
    global current_step, setup_complete; mode = mode_var.get(); print(f"[INFO] Mode changed to: {mode}"); reset_setup_vars(); setup_complete = False; current_step = "LOAD_VIDEO"
    if initial_frame is not None:
        if mode == "runner": status_var.set("Mode:Runner. Step 1/3: Click 0m point."); current_step = "SELECT_0M"
        else: status_var.set("Mode:Pixel. Step 1/1: Drag ROI."); current_step = "SELECT_ROI"; redraw_overlays()
    else: status_var.set(f"Mode:{'Runner' if mode == 'runner' else 'Pixel'}. Load video.")
    start_button.config(state=tk.DISABLED); save_button.config(state=tk.DISABLED)

def reset_setup_vars():
    global x_pix_0m, y_pix_0m, x_pix_50m, y_pix_50m, bbox, roi_start_point, selecting_roi, setup_complete, distance_offset, tracker, tracker_name, results_data
    x_pix_0m=None; y_pix_0m=None; x_pix_50m=None; y_pix_50m=None; bbox=None; roi_start_point=None; selecting_roi=False
    setup_complete=False; distance_offset=0.0; tracker=None; tracker_name=""; results_data=[]
    print("[DEBUG] Setup variables reset.")

def load_video():
    global video_path, initial_frame, current_step, video_fps
    fpath = filedialog.askopenfilename(title="動画選択", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")]);
    if not fpath: return
    video_path = fpath; filepath_var.set(f"読込中..."); status_var.set("動画読込中..."); root.update_idletasks()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): messagebox.showerror("エラー", f"動画開けません:\n{video_path}"); filepath_var.set("動画未読込"); status_var.set("準備完了"); return
    fps = cap.get(cv2.CAP_PROP_FPS); video_fps = fps if fps and fps > 0 else DEFAULT_FPS_IF_MISSING; print(f"[INFO] FPS: {video_fps:.2f}")
    ret, frame = cap.read(); cap.release()
    if not ret: messagebox.showerror("エラー", f"フレーム読込失敗:\n{video_path}"); filepath_var.set("動画未読込"); status_var.set("準備完了"); return
    initial_frame = frame; filepath_var.set(f"読込完了: {video_path}"); reset_setup_vars()
    mode = mode_var.get()
    if mode == "runner": current_step = "SELECT_0M"; status_var.set("Step 1/3: Click 0m point.")
    else: current_step = "SELECT_ROI"; status_var.set("Step 1/1: Drag ROI.")
    canvas.update_idletasks(); resize_and_show_frame(canvas.winfo_width(), canvas.winfo_height())
    start_button.config(state=tk.DISABLED); save_button.config(state=tk.DISABLED)

def resize_and_show_frame(canvas_w, canvas_h):
    global resized_frame, display_image, tk_image_ref, scale_factor, offset_x, offset_y
    if initial_frame is None or canvas_w <= 1 or canvas_h <= 1: return
    frame_h, frame_w = initial_frame.shape[:2]; scale_w = canvas_w / frame_w; scale_h = canvas_h / frame_h; scale_factor = min(scale_w, scale_h)
    new_w = int(frame_w * scale_factor); new_h = int(frame_h * scale_factor)
    if new_w <= 0 or new_h <= 0: return
    resized_frame = cv2.resize(initial_frame, (new_w, new_h), interpolation=cv2.INTER_AREA); offset_x = (canvas_w - new_w) // 2; offset_y = (canvas_h - new_h) // 2
    show_frame_on_canvas(resized_frame); redraw_overlays()

def show_frame_on_canvas(frame_to_show):
    global display_image, tk_image_ref
    try: img_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB); img_pil = Image.fromarray(img_rgb); display_image = ImageTk.PhotoImage(img_pil); tk_image_ref = display_image; canvas.delete("all"); canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=display_image, tags="frame")
    except Exception as e: print(f"[ERROR] Failed to show frame: {e}")

def redraw_overlays():
    if resized_frame is None: return
    canvas.delete("overlays"); mode = mode_var.get()
    if mode == "runner":
        if x_pix_0m is not None and y_pix_0m is not None: disp_x, disp_y = original_to_display_coords(x_pix_0m, y_pix_0m); canvas.create_oval(disp_x-4, disp_y-4, disp_x+4, disp_y+4, fill="lime", outline="lime", tags="overlays"); canvas.create_text(disp_x + 10, disp_y, text="0m", fill="lime", anchor=tk.W, tags="overlays")
        if x_pix_50m is not None and y_pix_50m is not None: disp_x, disp_y = original_to_display_coords(x_pix_50m, y_pix_50m); canvas.create_oval(disp_x-4, disp_y-4, disp_x+4, disp_y+4, fill="red", outline="red", tags="overlays"); canvas.create_text(disp_x + 10, disp_y, text="50m", fill="red", anchor=tk.W, tags="overlays")
    if bbox is not None: x1, y1, w, h = bbox; disp_x1, disp_y1 = original_to_display_coords(x1, y1); disp_x2, disp_y2 = original_to_display_coords(x1 + w, y1 + h); canvas.create_rectangle(disp_x1, disp_y1, disp_x2, disp_y2, outline="blue", width=2, tags="overlays")

def display_to_original_coords(display_x, display_y):
    if scale_factor == 0 or initial_frame is None: return None, None
    original_x = (display_x - offset_x) / scale_factor; original_y = (display_y - offset_y) / scale_factor; h, w = initial_frame.shape[:2]; original_x = np.clip(original_x, 0, w - 1); original_y = np.clip(original_y, 0, h - 1); return int(original_x), int(original_y)

def original_to_display_coords(original_x, original_y):
    if scale_factor == 0: return None, None
    display_x = int(original_x * scale_factor + offset_x); display_y = int(original_y * scale_factor + offset_y); return display_x, display_y

def on_canvas_click(event):
    global current_step, x_pix_0m, y_pix_0m, x_pix_50m, y_pix_50m, roi_start_point, selecting_roi, bbox, setup_complete
    if initial_frame is None or current_step == "TRACKING": return
    disp_x, disp_y = event.x, event.y; orig_x, orig_y = display_to_original_coords(disp_x, disp_y); mode = mode_var.get(); print(f"\n[DEBUG] Click! Mode:{mode}, Step:{current_step}, Coords(Orig):({orig_x},{orig_y})")
    if orig_x is None: return
    if mode == "runner":
        if current_step == "SELECT_0M": x_pix_0m = orig_x; y_pix_0m = orig_y; status_var.set("Step 2/3: Click 50m point."); current_step = "SELECT_50M"; redraw_overlays()
        elif current_step == "SELECT_50M": x_pix_50m = orig_x; y_pix_50m = orig_y; status_var.set("Step 3/3: Drag ROI."); current_step = "SELECT_ROI"; redraw_overlays()
        elif current_step == "SELECT_ROI": selecting_roi = True; roi_start_point = (disp_x, disp_y); bbox = None; setup_complete = False; status_var.set("ROI selecting..."); start_button.config(state=tk.DISABLED); redraw_overlays()
    else:
         if current_step == "SELECT_ROI": selecting_roi = True; roi_start_point = (disp_x, disp_y); bbox = None; setup_complete = False; status_var.set("ROI selecting..."); start_button.config(state=tk.DISABLED); redraw_overlays()

def on_canvas_drag(event):
    if not selecting_roi or roi_start_point is None: return
    disp_x, disp_y = event.x, event.y; canvas.delete("feedback_roi"); canvas.create_rectangle(roi_start_point[0], roi_start_point[1], disp_x, disp_y, outline="lime", width=2, tags="feedback_roi")

def on_canvas_release(event):
    global selecting_roi, bbox, setup_complete
    if not selecting_roi or roi_start_point is None: return
    selecting_roi = False; canvas.delete("feedback_roi"); disp_x1, disp_y1 = roi_start_point; disp_x2, disp_y2 = event.x, event.y; orig_x1, orig_y1 = display_to_original_coords(disp_x1, disp_y1); orig_x2, orig_y2 = display_to_original_coords(disp_x2, disp_y2)
    if orig_x1 is None or orig_x2 is None: messagebox.showerror("エラー", "ROI座標変換失敗。"); return
    w=abs(orig_x1-orig_x2); h=abs(orig_y1-orig_y2); roi_x=min(orig_x1,orig_x2); roi_y=min(orig_y1,orig_y2); print(f"[DEBUG] ROI Release. Orig coords: ({roi_x}, {roi_y}, {w}, {h})")
    if w < MIN_ROI_WIDTH or h < MIN_ROI_HEIGHT: messagebox.showwarning("ROIエラー", f"選択領域小さすぎ。最低{MIN_ROI_WIDTH}x{MIN_ROI_HEIGHT}必要。"); bbox=None; setup_complete=False; status_var.set("ROI選択失敗。再度ドラッグ。"); start_button.config(state=tk.DISABLED)
    else: bbox=(int(roi_x), int(roi_y), int(w), int(h)); setup_complete=True; status_var.set(f"設定完了。ROI={bbox}。開始可。"); start_button.config(state=tk.NORMAL)
    redraw_overlays()

def on_canvas_configure(event):
    print(f"[DEBUG] Canvas configured to: {event.width} x {event.height}");
    if initial_frame is not None and current_step != "TRACKING": resize_and_show_frame(event.width, event.height)

# --- トラッキングワーカースレッド ---
def tracking_worker():
    global tracker, tracker_name, results_data, current_step, distance_offset, video_fps, tracking_queue, stop_event
    mode = mode_var.get(); print("[Thread] Worker thread started.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): tracking_queue.put({"status": "エラー: 動画を開けません。", "error": True}); return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); frame_number = -1; total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); thread_results_data = []
    while True:
        if stop_event.is_set(): print("[Thread] Stop event received."); break
        frame_number += 1; ret, frame = cap.read()
        if not ret: print("[Thread] End of video."); break
        try: success, new_bbox_float = tracker.update(frame)
        except Exception as e_update: print(f"[Thread][ERROR] tracker.update failed: {e_update}"); success = False
        target_found = success; current_bbox_int = None; pixel_x_center = None; pixel_y_center = None; distance_m = None; raw_distance_m = None
        current_time_sec = frame_number / video_fps if video_fps > 0 else 0.0
        frame_display = frame.copy()
        if success:
            current_bbox_int = tuple(map(int, new_bbox_float)); pixel_x_center = current_bbox_int[0] + current_bbox_int[2] / 2; pixel_y_center = current_bbox_int[1] + current_bbox_int[3] / 2
            p1 = (current_bbox_int[0], current_bbox_int[1]); p2 = (current_bbox_int[0] + current_bbox_int[2], current_bbox_int[1] + current_bbox_int[3]); cv2.rectangle(frame_display, p1, p2, (0, 255, 0), 2, 1); label = f"{tracker_name}"
            if mode == "runner":
                raw_distance_m = 50.0 * (pixel_x_center - x_pix_0m) / (x_pix_50m - x_pix_0m); distance_m = raw_distance_m + distance_offset
                if distance_m is not None: label += f" D:{distance_m:.2f}m"; cv2.circle(frame_display, (int(pixel_x_center), p2[1]), 5, (0, 255, 0), -1)
            else: label += f" X:{pixel_x_center:.0f} Y:{pixel_y_center:.0f}"; cv2.circle(frame_display, (int(pixel_x_center), int(pixel_y_center)), 5, (0, 255, 0), -1)
            cv2.putText(frame_display, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else: cv2.putText(frame_display, f"{tracker_name} Lost", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        result_row = {'frame': frame_number, 'time_sec': current_time_sec, 'success': target_found, 'tracker': tracker_name, 'bbox_x': current_bbox_int[0] if target_found else None, 'bbox_y': current_bbox_int[1] if target_found else None, 'bbox_w': current_bbox_int[2] if target_found else None, 'bbox_h': current_bbox_int[3] if target_found else None,}
        if mode == "runner": result_row['pixel_x_center'] = pixel_x_center if target_found else None; result_row['raw_distance_m'] = raw_distance_m if target_found else None; result_row['distance_m'] = distance_m if target_found else None
        else: result_row['pixel_x_center'] = pixel_x_center if target_found else None; result_row['pixel_y_center'] = pixel_y_center if target_found else None
        thread_results_data.append(result_row)

        # --- ★★★ ここを修正 ★★★ ---
        try:
             status = f"処理中: {frame_number}/{total_frames}"
             tracking_queue.put({
                 "frame_display": frame_display,
                 "frame_number": frame_number,
                 "total_frames": total_frames,
                 "status": status,
                 "error": False
             }, block=False) # block=False でキューが満杯ならエラー
        except queue.Full:
             # キューが満杯の場合は何もしない (pass)
             pass
        except Exception as e_queue:
             # その他のキュー関連エラー
             print(f"[Thread][ERROR] Queue put failed: {e_queue}")
        # --------------------------

    cap.release()
    tracking_queue.put({"status": f"完了。{len(thread_results_data)}フレーム処理。", "error": False, "finished": True, "results": thread_results_data})
    print("[Thread] Worker thread finished.")

# --- GUI更新用関数 ---
def check_tracking_queue():
    global tk_image_ref, results_data
    try:
        data = tracking_queue.get(block=False)
        if "status" in data: status_var.set(data["status"])
        if data.get("frame_display") is not None:
            frame_to_display = data["frame_display"]; canvas_w = canvas.winfo_width(); canvas_h = canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                frame_h, frame_w = frame_to_display.shape[:2]; scale_w = canvas_w / frame_w; scale_h = canvas_h / frame_h; current_scale = min(scale_w, scale_h)
                new_w = int(frame_w * current_scale); new_h = int(frame_h * current_scale)
                if new_w > 0 and new_h > 0:
                     resized_display = cv2.resize(frame_to_display, (new_w, new_h), interpolation=cv2.INTER_AREA)
                     current_offset_x = (canvas_w - new_w) // 2; current_offset_y = (canvas_h - new_h) // 2
                     img_rgb = cv2.cvtColor(resized_display, cv2.COLOR_BGR2RGB); img_pil = Image.fromarray(img_rgb); current_display_image = ImageTk.PhotoImage(img_pil); tk_image_ref = current_display_image
                     canvas.delete("all"); canvas.create_image(current_offset_x, current_offset_y, anchor=tk.NW, image=current_display_image, tags="frame")
        if "frame_number" in data and "total_frames" in data: progress_var.set(data["frame_number"]); progressbar['maximum'] = data["total_frames"]
        if data.get("finished") or data.get("error"):
            print("[GUI] Received finished/error signal.");
            if data.get("results"): results_data = data["results"]; print(f"[GUI] Received {len(results_data)} results.")
            stop_tracking(from_worker=True)
            if data.get("error"): messagebox.showerror("エラー", data.get("status", "不明なエラー"))
            elif results_data: save_button.config(state=tk.NORMAL);
            if results_data and not data.get("error"): # ★ 正常完了の場合のみダイアログ表示
                if messagebox.askyesno("グラフ表示", "完了しました。\n結果をグラフ表示しますか？"): prepare_and_show_graph()
            return
        root.after(PREVIEW_UPDATE_INTERVAL_MS, check_tracking_queue)
    except queue.Empty:
        if tracking_thread and tracking_thread.is_alive(): root.after(PREVIEW_UPDATE_INTERVAL_MS, check_tracking_queue)
        else: print("[GUI] Worker stopped or queue empty."); stop_tracking(from_worker=True)
    except Exception as e: print(f"[ERROR] Error in check_tracking_queue: {e}"); traceback.print_exc(); stop_tracking(from_worker=True)

# --- トラッキング開始/停止関数 ---
def start_tracking_threaded():
    global tracker, tracker_name, results_data, current_step, distance_offset, tracking_thread, stop_event; mode = mode_var.get()
    if mode == "runner" and (not setup_complete or bbox is None or x_pix_0m is None or x_pix_50m is None): messagebox.showerror("エラー", "Runner: 設定未完了。"); return
    if mode == "runner" and x_pix_0m == x_pix_50m: messagebox.showerror("エラー", "Runner: 0m/50m X座標同じ。"); return
    if mode == "pixel" and (not setup_complete or bbox is None): messagebox.showerror("エラー", "Pixel: 設定未完了。"); return
    if initial_frame is None: messagebox.showerror("エラー", "動画未読込。"); return
    status_var.set("初期化中..."); load_button.config(state=tk.DISABLED); start_button.config(state=tk.DISABLED); stop_button.config(state=tk.DISABLED); save_button.config(state=tk.DISABLED); root.update_idletasks()
    tracker_obj, name, success = try_init_tracker("CSRT", initial_frame, bbox)
    if not success: print("CSRT失敗、KCF試行..."); tracker_obj, name, success = try_init_tracker("KCF", initial_frame, bbox)
    if not success: messagebox.showerror("初期化エラー", "初期化失敗。\nログ確認、ROI変更など試してください。"); status_var.set("初期化失敗"); load_button.config(state=tk.NORMAL); return
    tracker = tracker_obj; tracker_name = name; distance_offset = 0.0
    if mode == "runner": initial_target_x = bbox[0] + bbox[2] / 2; raw_initial_distance = 50.0 * (initial_target_x - x_pix_0m) / (x_pix_50m - x_pix_0m);
    if raw_initial_distance > 0: distance_offset = -raw_initial_distance; print(f"[INFO] Runner Offset {distance_offset:.2f}m applied.")
    else: print(f"[INFO] Runner No offset (Initial: {raw_initial_distance:.2f}m).")
    results_data = []; stop_event.clear()
    while not tracking_queue.empty():
        try: tracking_queue.get_nowait()
        except queue.Empty: break
    tracking_thread = threading.Thread(target=tracking_worker, daemon=True); tracking_thread.start()
    current_step = "TRACKING"; status_var.set(f"トラッキング開始 ({tracker_name})..."); load_button.config(state=tk.DISABLED); start_button.config(state=tk.DISABLED); stop_button.config(state=tk.NORMAL); save_button.config(state=tk.DISABLED)
    root.after(PREVIEW_UPDATE_INTERVAL_MS, check_tracking_queue)

def stop_tracking(from_worker=False):
    global current_step, tracking_thread
    if not from_worker and tracking_thread and tracking_thread.is_alive(): print("[GUI] Stop button pressed."); stop_event.set()
    else: print("[GUI] Worker finished/stopped or no active thread.")
    load_button.config(state=tk.NORMAL); start_button.config(state=tk.NORMAL); stop_button.config(state=tk.DISABLED);
    if results_data: save_button.config(state=tk.NORMAL)
    mode = mode_var.get(); current_step = "SELECT_0M" if mode == "runner" else "SELECT_ROI"


# --- データ処理＆グラフ表示関連関数 ---

# ★★★ 加速度の移動平均計算を追加 ★★★
def calculate_derivatives_and_averages(df, mode, interval_seconds, fps):
    if df.empty: return df
    print("[INFO] Calculating derivatives and moving averages...")
    window_size = max(1, int(interval_seconds * fps))
    print(f"[INFO] Moving average window: {window_size} frames")
    df['time_diff'] = df['time_sec'].diff().fillna(1/fps)
    df.loc[df['time_diff'] <= 1e-6, 'time_diff'] = 1/fps
    avg_suffix = f'_avg_{interval_seconds:.1f}s'

    if mode == 'runner':
        cols = {'pos': 'distance_m', 'vel': 'velocity_m_s', 'accel': 'acceleration_m_s2'}
        numeric_pos = pd.to_numeric(df[cols['pos']], errors='coerce')
        df[cols['pos'] + avg_suffix] = numeric_pos.rolling(window=window_size, center=True, min_periods=1).mean()
        df[cols['vel']] = numeric_pos.diff() / df['time_diff']
        numeric_vel = pd.to_numeric(df[cols['vel']], errors='coerce')
        df[cols['vel'] + avg_suffix] = numeric_vel.rolling(window=window_size, center=True, min_periods=1).mean()
        df[cols['accel']] = numeric_vel.diff() / df['time_diff'] # 生の加速度
        numeric_accel = pd.to_numeric(df[cols['accel']], errors='coerce')
        df[cols['accel'] + avg_suffix] = numeric_accel.rolling(window=window_size, center=True, min_periods=1).mean() # 加速度の移動平均
        print(f"[DEBUG] Calculated runner cols: {[c for c in cols.values()]+[c+avg_suffix for c in cols.values()]}")

    else: # pixelモード
        for axis in ['x', 'y']:
            pos_col = f'pixel_{axis}_center'; vel_col = f'velocity_{axis}_px_s'; accel_col = f'acceleration_{axis}_px_s2'
            print(f"[DEBUG] Calculating pixel cols (axis={axis}): {pos_col}, {vel_col}, {accel_col}")
            numeric_pos = pd.to_numeric(df[pos_col], errors='coerce')
            df[pos_col + avg_suffix] = numeric_pos.rolling(window=window_size, center=True, min_periods=1).mean()
            df[vel_col] = numeric_pos.diff() / df['time_diff']
            numeric_vel = pd.to_numeric(df[vel_col], errors='coerce')
            df[vel_col + avg_suffix] = numeric_vel.rolling(window=window_size, center=True, min_periods=1).mean()
            df[accel_col] = numeric_vel.diff() / df['time_diff'] # 生の加速度
            numeric_accel = pd.to_numeric(df[accel_col], errors='coerce')
            df[accel_col + avg_suffix] = numeric_accel.rolling(window=window_size, center=True, min_periods=1).mean() # 加速度の移動平均

    if 'time_diff' in df.columns: df = df.drop(columns=['time_diff'])
    print("[INFO] Calculations complete.")
    return df

def prepare_and_show_graph():
    global graph_df
    if not results_data: messagebox.showwarning("グラフエラー", "データなし。"); return
    try: interval_sec_str = moving_avg_sec_var.get(); interval_seconds = float(interval_sec_str if interval_sec_str else DEFAULT_MOVING_AVG_SEC)
    except ValueError: messagebox.showerror("入力エラー", f"移動平均区間に数値入力要。"); return
    if interval_seconds <= 0: interval_seconds = DEFAULT_MOVING_AVG_SEC
    df_temp = pd.DataFrame(results_data); mode = mode_var.get()
    try: graph_df = calculate_derivatives_and_averages(df_temp, mode, interval_seconds, video_fps)
    except Exception as e_calc: messagebox.showerror("計算エラー", f"速度等計算エラー:\n{e_calc}"); traceback.print_exc(); return
    if graph_df is None or graph_df.empty: messagebox.showerror("計算エラー", "データ処理失敗。"); return
    create_graph_window(graph_df, mode, interval_seconds)

# ★★★ グラフウィンドウ作成 (Y軸自動スケール調整追加) ★★★
def create_graph_window(df, mode, interval_s):
    global graph_window
    if graph_window is not None and graph_window.winfo_exists(): graph_window.destroy()
    graph_window = tk.Toplevel(root); graph_window.title("結果グラフ"); graph_window.geometry("1000x700")

    fig = Figure(figsize=(10, 6), dpi=100); ax1 = fig.add_subplot(111); ax1.set_xlabel("Time (s)")
    ax2 = ax1.twinx(); ax3 = ax1.twinx(); ax3.spines["right"].set_position(("axes", 1.1)); fig.subplots_adjust(right=0.8)
    colors = {'pos': 'blue', 'vel': 'green', 'accel': 'red'}
    avg_suffix = f'_avg_{interval_s:.1f}s'

    # Y軸ラベル設定
    ax1.set_ylabel("Position (m or px)", color=colors['pos']); ax1.tick_params(axis='y', labelcolor=colors['pos'])
    ax2.set_ylabel("Velocity (m/s or px/s)", color=colors['vel']); ax2.tick_params(axis='y', labelcolor=colors['vel'])
    ax3.set_ylabel("Acceleration (m/s² or px/s²)", color=colors['accel']); ax3.tick_params(axis='y', labelcolor=colors['accel'])

    time_col = 'time_sec'
    plot_data_exists = False # プロットするデータが何かあるか

    if mode == 'runner':
        pos_col = 'distance_m'; pos_avg_col = pos_col + avg_suffix
        vel_col = 'velocity_m_s'; vel_avg_col = vel_col + avg_suffix
        accel_col = 'acceleration_m_s2'; accel_avg_col = accel_col + avg_suffix

        if pos_col in df: ax1.plot(df[time_col], df[pos_col], color=colors['pos'], linestyle=':', linewidth=1, label='Dist (Raw)'); plot_data_exists=True
        if pos_avg_col in df: ax1.plot(df[time_col], df[pos_avg_col], color=colors['pos'], linestyle='-', linewidth=2, label=f'Dist ({interval_s:.1f}s Avg)'); plot_data_exists=True
        if vel_col in df: ax2.plot(df[time_col], df[vel_col], color=colors['vel'], linestyle=':', linewidth=1, label='Vel (Raw)'); plot_data_exists=True
        if vel_avg_col in df: ax2.plot(df[time_col], df[vel_avg_col], color=colors['vel'], linestyle='-', linewidth=2, label=f'Vel ({interval_s:.1f}s Avg)'); plot_data_exists=True
        if accel_col in df: ax3.plot(df[time_col], df[accel_col], color=colors['accel'], linestyle=':', linewidth=1, label='Accel (Raw)'); plot_data_exists=True
        if accel_avg_col in df: ax3.plot(df[time_col], df[accel_avg_col], color=colors['accel'], linestyle='-', linewidth=2, label=f'Accel ({interval_s:.1f}s Avg)'); plot_data_exists=True

    else: # pixelモード (X軸のみ表示例)
        # Y軸を表示する場合は、以下のロジックをY軸の列名でも繰り返す
        pos_col = 'pixel_x_center'; pos_avg_col = pos_col + avg_suffix
        vel_col = 'velocity_x_px_s'; vel_avg_col = vel_col + avg_suffix
        accel_col = 'acceleration_x_px_s2'; accel_avg_col = accel_col + avg_suffix

        if pos_col in df: ax1.plot(df[time_col], df[pos_col], color=colors['pos'], linestyle=':', linewidth=1, label='X Px (Raw)'); plot_data_exists=True
        if pos_avg_col in df: ax1.plot(df[time_col], df[pos_avg_col], color=colors['pos'], linestyle='-', linewidth=2, label=f'X Px ({interval_s:.1f}s Avg)'); plot_data_exists=True
        if vel_col in df: ax2.plot(df[time_col], df[vel_col], color=colors['vel'], linestyle=':', linewidth=1, label='X Vel (Raw)'); plot_data_exists=True
        if vel_avg_col in df: ax2.plot(df[time_col], df[vel_avg_col], color=colors['vel'], linestyle='-', linewidth=2, label=f'X Vel ({interval_s:.1f}s Avg)'); plot_data_exists=True
        if accel_col in df: ax3.plot(df[time_col], df[accel_col], color=colors['accel'], linestyle=':', linewidth=1, label='X Accel (Raw)'); plot_data_exists=True
        if accel_avg_col in df: ax3.plot(df[time_col], df[accel_avg_col], color=colors['accel'], linestyle='-', linewidth=2, label=f'X Accel ({interval_s:.1f}s Avg)'); plot_data_exists=True

    # --- ★★★ Y軸の範囲を移動平均に合わせて自動調整 ★★★ ---
    def set_axis_limits(ax, avg_column_name, df, padding_factor=0.05):
        if avg_column_name in df and not df[avg_column_name].isnull().all():
            min_val = df[avg_column_name].min()
            max_val = df[avg_column_name].max()
            data_range = max_val - min_val
            if data_range > 1e-6: # 範囲がほぼゼロでない場合
                padding = data_range * padding_factor
                lower = min_val - padding
                upper = max_val + padding
            else: # 範囲がゼロまたは非常に小さい場合 (一定値など)
                padding = abs(min_val * padding_factor) if abs(min_val) > 1e-6 else 1.0 # 値が0でなければ割合、0なら固定値
                lower = min_val - padding
                upper = max_val + padding
            ax.set_ylim(lower, upper)
            print(f"[DEBUG] Set Y limits for {avg_column_name}: ({lower:.2f}, {upper:.2f})")
        else:
             print(f"[DEBUG] Could not set Y limits for {avg_column_name} (column missing or all NaN).")

    if plot_data_exists: # 何かプロットされていたら範囲調整
        set_axis_limits(ax1, pos_avg_col, df)
        set_axis_limits(ax2, vel_avg_col, df)
        set_axis_limits(ax3, accel_avg_col, df)
        # ピクセルモードでY軸もプロットした場合は、対応するY軸の avg_col を使って同様に設定
        #例: set_axis_limits(ax1_for_y, y_pos_avg_col, df)
    # ------------------------------------------------------

    # 凡例
    lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels(); lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='best', fontsize='small'); ax1.grid(True); fig.tight_layout()

    # Tkinter埋め込み
    canvas_widget = FigureCanvasTkAgg(fig, master=graph_window); canvas_widget.draw()
    canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    toolbar = NavigationToolbar2Tk(canvas_widget, graph_window); toolbar.update()
    canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # 保存ボタンのみ
    button_frame = ttk.Frame(graph_window); button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    ttk.Button(button_frame, text="グラフを画像保存", command=lambda: save_graph_image(fig)).pack(side=tk.RIGHT, padx=5)

def save_graph_image(fig): # (変更なし)
    if fig is None: return
    save_path = filedialog.asksaveasfilename(title="グラフ画像保存", defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf"), ("All", "*.*")])
    if not save_path: return
    try: fig.savefig(save_path, dpi=150); messagebox.showinfo("保存完了", f"保存しました:\n{save_path}")
    except Exception as e: messagebox.showerror("保存エラー", f"エラー:\n{e}")

# ★ CSV保存関数 (列名リスト更新) ★
def save_csv():
    if not results_data: messagebox.showwarning("保存エラー", "データがありません。"); return
    mode = mode_var.get(); initial_file = f"tracking_results_{mode}.csv"; interval_seconds = DEFAULT_MOVING_AVG_SEC
    try: interval_seconds_str = moving_avg_sec_var.get(); interval_seconds = float(interval_seconds_str if interval_seconds_str else DEFAULT_MOVING_AVG_SEC)
    except ValueError: print(f"[WARN] Invalid interval input. Using default."); interval_seconds = DEFAULT_MOVING_AVG_SEC
    if interval_seconds <= 0: interval_seconds = DEFAULT_MOVING_AVG_SEC
    save_path = filedialog.asksaveasfilename(title="CSV保存", defaultextension=".csv", initialfile=initial_file, filetypes=[("CSV", "*.csv"), ("All", "*.*")])
    if not save_path: return
    try:
        df_save = pd.DataFrame(results_data)
        df_save = calculate_derivatives_and_averages(df_save, mode, interval_seconds, video_fps)
        if df_save is None or df_save.empty: raise ValueError("Data processing failed.")
        cols_common = ['frame', 'time_sec', 'success', 'tracker', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']
        avg_suffix = f'_avg_{interval_seconds:.1f}s'
        # ★ 更新された列名リスト (加速度の生データも含む) ★
        if mode == 'runner':
            cols_mode = ['pixel_x_center', 'raw_distance_m', 'distance_m', f'distance_m{avg_suffix}',
                         'velocity_m_s', f'velocity_m_s{avg_suffix}',
                         'acceleration_m_s2', f'acceleration_m_s2{avg_suffix}']
        else: # pixel モード (X,Y両方)
            cols_mode = ['pixel_x_center', f'pixel_x_center{avg_suffix}', 'pixel_y_center', f'pixel_y_center{avg_suffix}',
                         'velocity_x_px_s', f'velocity_x_px_s{avg_suffix}', 'acceleration_x_px_s2', f'acceleration_x_px_s2{avg_suffix}',
                         'velocity_y_px_s', f'velocity_y_px_s{avg_suffix}', 'acceleration_y_px_s2', f'acceleration_y_px_s2{avg_suffix}']
        all_cols = cols_common + cols_mode; df_ordered = df_save[[col for col in all_cols if col in df_save.columns]]
        df_ordered.to_csv(save_path, index=False, encoding='utf-8-sig'); messagebox.showinfo("完了", f"保存しました:\n{save_path}"); status_var.set(f"CSV保存完了: {save_path}")
    except Exception as e: messagebox.showerror("保存/計算エラー", f"エラー:\n{e}"); status_var.set("CSV保存/計算エラー")


# --- アプリ終了時の処理 (変更なし) ---
def on_closing(): print("Window closing..."); stop_event.set(); root.destroy()

# --- GUIセットアップ (ttk, 変更なし) ---
root = tk.Tk(); root.title("動画トラッカー"); root.geometry("900x700"); root.protocol("WM_DELETE_WINDOW", on_closing)
setup_frame = ttk.Frame(root, padding="5 5 5 5"); setup_frame.pack(side=tk.TOP, fill=tk.X)
tk.Label(setup_frame, text="追跡モード:").pack(side=tk.LEFT, padx=(0,5)); mode_var = tk.StringVar(value="runner"); mode_var.trace_add("write", update_ui_for_mode)
runner_radio = ttk.Radiobutton(setup_frame, text="ランナー追跡", variable=mode_var, value="runner"); runner_radio.pack(side=tk.LEFT)
pixel_radio = ttk.Radiobutton(setup_frame, text="ピクセル追跡", variable=mode_var, value="pixel"); pixel_radio.pack(side=tk.LEFT, padx=(0, 15))
ttk.Label(setup_frame, text="移動平均区間(秒):").pack(side=tk.LEFT); moving_avg_sec_var = tk.StringVar(value=str(DEFAULT_MOVING_AVG_SEC)); moving_avg_entry = ttk.Entry(setup_frame, textvariable=moving_avg_sec_var, width=5); moving_avg_entry.pack(side=tk.LEFT)
top_frame = ttk.Frame(root, padding="5 0 5 5"); top_frame.pack(side=tk.TOP, fill=tk.X)
load_button = ttk.Button(top_frame, text="動画ファイルを開く", command=load_video); load_button.pack(side=tk.LEFT)
filepath_var = tk.StringVar(value="動画未読込"); filepath_frame = ttk.Frame(top_frame, relief=tk.SUNKEN, borderwidth=1); filepath_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5); filepath_label = ttk.Label(filepath_frame, textvariable=filepath_var, anchor="w"); filepath_label.pack(fill=tk.X, padx=2, pady=2)
canvas_frame = ttk.Frame(root, relief=tk.SUNKEN, borderwidth=1); canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
canvas = tk.Canvas(canvas_frame, bg="black"); canvas.pack(fill=tk.BOTH, expand=True); canvas.bind("<Configure>", on_canvas_configure); canvas.bind("<Button-1>", on_canvas_click); canvas.bind("<B1-Motion>", on_canvas_drag); canvas.bind("<ButtonRelease-1>", on_canvas_release)
bottom_frame = ttk.Frame(root, padding="5 5 5 5"); bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
start_button = ttk.Button(bottom_frame, text="開始", command=start_tracking_threaded, state=tk.DISABLED); start_button.pack(side=tk.LEFT, padx=(0,5))
stop_button = ttk.Button(bottom_frame, text="停止", command=stop_tracking, state=tk.DISABLED); stop_button.pack(side=tk.LEFT, padx=5)
save_button = ttk.Button(bottom_frame, text="CSV保存", command=save_csv, state=tk.DISABLED); save_button.pack(side=tk.LEFT, padx=5)
status_var = tk.StringVar(value="モードを選択し、動画ファイルを開いてください。"); status_label = ttk.Label(bottom_frame, textvariable=status_var, anchor="w"); status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
progress_var = tk.DoubleVar(); progressbar = ttk.Progressbar(bottom_frame, variable=progress_var, orient="horizontal", length=200, mode="determinate"); progressbar.pack(side=tk.RIGHT, padx=5)
update_ui_for_mode(); root.mainloop()