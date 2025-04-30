# -*- coding: utf-8 -*-
import cv2
import sys
import platform
import pandas as pd
import numpy as np # 念のため
import traceback # 例外の詳細表示用

# --- 設定 ---
VIDEO_PATH = 'your_video.mp4' # ★ 対象の動画ファイルパスに変更してください
OUTPUT_CSV = 'tracking_results_opencv_tracker_nonetest.csv' # ★ 出力CSVファイル名 (テスト用)
MIN_ROI_WIDTH = 20
MIN_ROI_HEIGHT = 20

# --- グローバル変数 ---
x_pix_0m = None; x_pix_50m = None
bbox = None; selecting_roi = False; roi_start_point = None
initial_frame = None; tracker = None; tracker_name = ""
current_step = "SELECT_0M"; setup_complete = False
results_data = []

# --- マウスコールバック関数 (設定用) ---
def mouse_callback_setup(event, x, y, flags, param):
    global current_step, x_pix_0m, x_pix_50m, bbox, selecting_roi, roi_start_point, setup_complete, initial_frame

    if current_step == "TRACKING": return

    frame_display = param.get('frame_display', initial_frame)
    if frame_display is None: return

    frame_height, frame_width = frame_display.shape[:2]
    if not (0 <= x < frame_width and 0 <= y < frame_height): return

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_step == "SELECT_0M":
             x_pix_0m = x; print(f"[DEBUG] 0m X set to: {x_pix_0m}")
             temp_frame = frame_display.copy(); cv2.circle(temp_frame, (x, y), 5, (0, 255, 0), -1); cv2.putText(temp_frame, "0m", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
             cv2.imshow("Setup Tracking", temp_frame); print("次に、50m地点をクリックしてください。"); current_step = "SELECT_50M"
        elif current_step == "SELECT_50M":
            x_pix_50m = x; print(f"[DEBUG] 50m X set to: {x_pix_50m}")
            if x_pix_0m is not None:
                if abs(x_pix_0m - x_pix_50m) < 10: print("警告: 0mと50mが近すぎます！")
                elif x_pix_0m > x_pix_50m: print("警告: 0mが50mより右側です！")
            temp_frame = frame_display.copy()
            if x_pix_0m is not None: cv2.circle(temp_frame, (x_pix_0m, y), 5, (0, 255, 0), -1); cv2.putText(temp_frame, "0m", (x_pix_0m+10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            cv2.circle(temp_frame, (x, y), 5, (0, 0, 255), -1); cv2.putText(temp_frame, "50m", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Setup Tracking", temp_frame); print("次に、追跡したい対象をドラッグして選択してください。"); current_step = "SELECT_ROI"
        elif current_step == "SELECT_ROI":
            selecting_roi = True; roi_start_point = (x, y); bbox = None; setup_complete = False
            print("[DEBUG] Starting ROI selection drag.")
            temp_frame = initial_frame.copy()
            h, w = temp_frame.shape[:2]
            if x_pix_0m is not None: cv2.circle(temp_frame, (x_pix_0m, h//2), 5, (0,255,0), -1); cv2.putText(temp_frame, "0m", (x_pix_0m+10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            if x_pix_50m is not None: cv2.circle(temp_frame, (x_pix_50m, h//2), 5, (0,0,255), -1); cv2.putText(temp_frame, "50m", (x_pix_50m+10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.imshow("Setup Tracking", temp_frame)

    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting_roi:
            temp_frame = frame_display.copy(); cv2.rectangle(temp_frame, roi_start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow("Setup Tracking", temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        if selecting_roi and roi_start_point is not None:
            selecting_roi = False
            x1, y1 = roi_start_point; x2, y2 = x, y
            w = abs(x1 - x2); h = abs(y1 - y2)
            roi_x = min(x1, x2); roi_y = min(y1, y2)
            print(f"[DEBUG] ROI selection ended. Raw coords: ({roi_x}, {roi_y}, {w}, {h})")

            if w < MIN_ROI_WIDTH or h < MIN_ROI_HEIGHT:
                print(f"エラー: 選択領域が小さすぎます (幅:{w}, 高さ:{h})。最低{MIN_ROI_WIDTH}x{MIN_ROI_HEIGHT}必要です。")
                bbox = None; setup_complete = False
                temp_frame = initial_frame.copy()
                fh, fw = temp_frame.shape[:2]
                if x_pix_0m is not None: cv2.circle(temp_frame, (x_pix_0m, fh//2), 5, (0,255,0), -1); cv2.putText(temp_frame, "0m", (x_pix_0m+10, fh//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
                if x_pix_50m is not None: cv2.circle(temp_frame, (x_pix_50m, fh//2), 5, (0,0,255), -1); cv2.putText(temp_frame, "50m", (x_pix_50m+10, fh//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                cv2.imshow("Setup Tracking", temp_frame)
            else:
                bbox = (int(roi_x), int(roi_y), int(w), int(h))
                print(f"[DEBUG] ROI selected and stored as integer tuple: {bbox}, Type: {type(bbox)}")
                print(f"[DEBUG]   Element types: {[type(el) for el in bbox]}")
                setup_complete = True
                temp_frame = frame_display.copy()
                cv2.rectangle(temp_frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
                cv2.imshow("Setup Tracking", temp_frame)
                print("設定完了。Enterキーを押してトラッキングを開始してください。")

# --- トラッカー初期化試行関数 (None を成功扱いにする修正込み) ---
def try_init_tracker(tracker_type_str, frame, roi):
    tracker_obj = None
    success = False
    actual_name = ""
    print(f"\n--- Trying to initialize {tracker_type_str} tracker ---")
    print(f"[DEBUG] Input frame shape: {frame.shape}, dtype: {frame.dtype}")
    print(f"[DEBUG] Input ROI value: {roi}, type: {type(roi)}")
    if roi is None: print("[DEBUG] Error: ROI is None."); return None, "", False
    if not isinstance(roi, tuple) or len(roi) != 4 or not all(isinstance(n, int) for n in roi):
        print(f"[DEBUG] Error: ROI {roi} is not a valid integer tuple."); return None, "", False

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
                print(f"[DEBUG] Tracker object created successfully.")
                print(f"[DEBUG] Calling {actual_name}.init(frame, roi={roi})...")
                init_result = tracker_obj.init(frame, roi)
                print(f"[DEBUG] {actual_name}.init() returned: {init_result} (Type: {type(init_result)})")

                # --- ★★★ 対策 B: None の扱いを変更 ★★★ ---
                # 明示的に False が返らない限り成功と仮定 (None も成功扱い)
                success = (init_result is not False)
                if not success:
                     print(f"[DEBUG] Initialization explicitly failed (returned False).")
                elif init_result is None:
                     # Noneが返ってきたことを警告として表示
                     print(f"[DEBUG] WARNING: init() returned None. Assuming success for testing purposes, but this behavior is unusual and might indicate an issue.")
                # ------------------------------------------

            else:
                 print(f"[DEBUG] Error: Failed to create {actual_name} tracker object."); return None, actual_name, False
        else:
             print(f"[DEBUG] Error: Creator function for {tracker_type_str} not found."); return None, "", False

        # --- ここから下の評価も変更 ---
        if success:
            # Noneでもここに来るようになった
            if init_result is not None: # None でない場合は通常の成功メッセージ
                 print(f"{actual_name} initialized successfully.")
            # else: # None の場合の警告は上で出したのでここでは省略
            return tracker_obj, actual_name, True
        else:
            # 明示的にFalseが返された場合、またはオブジェクト作成失敗
            print(f"Failed to initialize {actual_name}.")
            return None, actual_name, False
        # ----------------------------

    except Exception as e:
        print(f"[DEBUG] !!! Exception during {actual_name if actual_name else tracker_type_str} initialization !!!")
        print(f"[DEBUG] Exception type: {type(e)}")
        print(f"[DEBUG] Exception message: {e}")
        print("[DEBUG] Traceback:")
        traceback.print_exc()
        return None, actual_name, False

# --- メイン処理 ---
print(f"OpenCV Version: {cv2.__version__}")
print(f"Platform: {platform.system()} {platform.release()}")
if hasattr(cv2, "TrackerCSRT_create") or (hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create")): print("CSRT tracker seems available.")
else: print("Warning: CSRT tracker not found.")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened(): print(f"エラー: 動画 '{VIDEO_PATH}' を開けません。"); sys.exit()

ret, frame = cap.read()
if not ret: print("エラー: 動画から最初のフレームを読み込めません。"); cap.release(); sys.exit()
initial_frame = frame.copy()
print(f"[DEBUG] Initial frame loaded. Shape: {initial_frame.shape}, dtype: {initial_frame.dtype}")

cv2.namedWindow("Setup Tracking", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Setup Tracking", mouse_callback_setup, {'frame_display': initial_frame.copy()})

print("--- 設定手順 ---"); print("1. 0m地点 -> 2. 50m地点 -> 3. ROI選択(ドラッグ) -> 4. Enterキー ('q'/'r'あり)")
cv2.imshow("Setup Tracking", initial_frame)

# --- 設定ループ ---
while current_step != "TRACKING":
    key = cv2.waitKey(1) & 0xFF

    if key == 13: # Enterキー
        if not setup_complete:
             print("エラー: 0m, 50m, ROI の全てを設定してください。")
        else:
            print("\nEnter pressed. Initializing tracker...")
            print(f"[DEBUG] Preparing to initialize tracker.")
            print(f"[DEBUG]   Using initial_frame shape: {initial_frame.shape}, dtype: {initial_frame.dtype}")
            print(f"[DEBUG]   Using bbox: {bbox}, type: {type(bbox)}")

            tracker_obj, name, success = try_init_tracker("CSRT", initial_frame, bbox)
            if not success:
                 print("\nCSRT failed. Trying KCF tracker...")
                 tracker_obj, name, success = try_init_tracker("KCF", initial_frame, bbox)

            if success:
                tracker = tracker_obj; tracker_name = name; current_step = "TRACKING"
                print(f"\nTracking will start using {tracker_name}.")
            else:
                print("\nエラー: トラッカーの初期化に最終的に失敗しました。")
                print("コンソールの [DEBUG] ログを確認してください。")
                print("'r'キーでリセットするか、'q'キーで終了してください。")

    elif key == ord('r'):
        print("\n[DEBUG] Resetting setup...")
        x_pix_0m = None; x_pix_50m = None; bbox = None
        selecting_roi = False; roi_start_point = None
        tracker = None; tracker_name = ""; current_step = "SELECT_0M"; setup_complete = False
        cv2.imshow("Setup Tracking", initial_frame)
        print("1. 0m地点をクリックしてください。")

    elif key == ord('q'):
        print("プログラムを終了します。"); cap.release(); cv2.destroyAllWindows(); sys.exit()

# --- トラッキングループ ---
if current_step == "TRACKING":
    print(f"Starting tracking loop with {tracker_name}...")
    cv2.destroyWindow("Setup Tracking")
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_number = -1

    while True:
        frame_number += 1
        ret, frame = cap.read()
        if not ret: print("\n動画の終わりに到達しました。"); break

        # トラッカー更新
        success, new_bbox_float = tracker.update(frame)

        target_found_in_frame = success
        current_x_pix = None; distance_m = None; current_bbox_int = None

        if success:
            current_bbox_int = tuple(map(int, new_bbox_float))
            current_x_pix = current_bbox_int[0] + current_bbox_int[2] / 2
            if x_pix_0m is not None and x_pix_50m is not None and x_pix_50m != x_pix_0m:
                distance_m = 50.0 * (current_x_pix - x_pix_0m) / (x_pix_50m - x_pix_0m)

        results_data.append({
            'frame': frame_number, 'success': target_found_in_frame, 'tracker': tracker_name,
            'pixel_x_bottom_center': current_x_pix if target_found_in_frame else None, 'distance_m': distance_m if target_found_in_frame else None,
            'bbox_x': current_bbox_int[0] if target_found_in_frame else None, 'bbox_y': current_bbox_int[1] if target_found_in_frame else None,
            'bbox_w': current_bbox_int[2] if target_found_in_frame else None, 'bbox_h': current_bbox_int[3] if target_found_in_frame else None,
        })

        frame_display = frame.copy()
        if target_found_in_frame:
            p1 = (current_bbox_int[0], current_bbox_int[1]); p2 = (current_bbox_int[0] + current_bbox_int[2], current_bbox_int[1] + current_bbox_int[3])
            cv2.rectangle(frame_display, p1, p2, (0, 255, 0), 2, 1)
            label = f"{tracker_name}"
            if distance_m is not None: label += f" Dist:{distance_m:.2f}m"
            cv2.putText(frame_display, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame_display, (int(current_x_pix), p2[1]), 5, (0, 255, 0), -1)
        else:
            cv2.putText(frame_display, f"{tracker_name} Tracking Failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.putText(frame_display, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if x_pix_0m is not None: cv2.line(frame_display, (int(x_pix_0m), 0), (int(x_pix_0m), frame_display.shape[0]), (0, 255, 0), 1)
        if x_pix_50m is not None: cv2.line(frame_display, (int(x_pix_50m), 0), (int(x_pix_50m), frame_display.shape[0]), (0, 0, 255), 1)

        cv2.imshow("Tracking", frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'): print("\nユーザー中断"); break

# --- 終了処理 & CSV保存 ---
cap.release(); cv2.destroyAllWindows()
print("トラッキング処理が完了しました。")
if results_data:
    df = pd.DataFrame(results_data)
    try: df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig'); print(f"結果を {OUTPUT_CSV} に保存しました。")
    except Exception as e: print(f"CSV保存エラー: {e}")
else: print("保存する結果データがありませんでした。")