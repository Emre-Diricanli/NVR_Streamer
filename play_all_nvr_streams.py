#!/usr/bin/env python3
"""
Show all NVR channels in ONE tiled window (grid mosaic).

- Discovers a working RTSP URL per channel using common templates.
- Starts one reader thread per channel (stores latest frame only).
- Renders a live grid window (e.g., 4x3 for 12 channels) with overlays.

Quit: press 'q' while the mosaic window is focused.

Requirements:
  - Python 3.8+
  - pip install opencv-python
  - (Recommended) FFmpeg installed on system
"""

import os
import time
import threading
from urllib.parse import quote
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np

# ----------------- CONFIGURE THESE -----------------
NVR_IP = "71.75.112.170"
NVR_PORT = 5541     # 554 is default; use your RTSP port
NVR_USER = "admin"
NVR_PASS = "Cstore2402"

CHANNEL_COUNT = 16    # number of channels to show
GRID_COLS = 4         # e.g., 4 columns x 3 rows = 12
GRID_ROWS = 4

# Overall mosaic resolution (width x height of the window)
CANVAS_W = 1600
CANVAS_H = 900

# Target FPS for the mosaic (reduce if your laptop struggles)
TARGET_FPS = 10

# Try both non-padded (1) and zero-padded (01) channel numbers with these templates
TRY_TEMPLATES = [
    # --- Hikvision (CCSS where CC=channel 01.., SS=01 main / 02 sub) ---
    "rtsp://{user}:{pwd}@{ip}:{port}/Streaming/Channels/{hik_cc}01",
    "rtsp://{user}:{pwd}@{ip}:{port}/Streaming/Channels/{hik_cc}02",

    # --- Dahua / Amcrest ---
    "rtsp://{user}:{pwd}@{ip}:{port}/cam/realmonitor?channel={ch}&subtype=0",
    "rtsp://{user}:{pwd}@{ip}:{port}/cam/realmonitor?channel={ch}&subtype=1",

    # --- Reolink ---
    "rtsp://{user}:{pwd}@{ip}:{port}/h264Preview_{ch}_main",
    "rtsp://{user}:{pwd}@{ip}:{port}/h264Preview_{ch}_sub",
    "rtsp://{user}:{pwd}@{ip}:{port}/Preview_{ch}_main",
    "rtsp://{user}:{pwd}@{ip}:{port}/Preview_{ch}_sub",

    # --- Other common patterns ---
    "rtsp://{user}:{pwd}@{ip}:{port}/h264/ch{ch}/main/av_stream",
    "rtsp://{user}:{pwd}@{ip}:{port}/ch{ch}/0",
    "rtsp://{user}:{pwd}@{ip}:{port}/live/ch{ch}",
    "rtsp://{user}:{pwd}@{ip}:{port}/media/video{ch}",

    # --- Night Owl Protect (some models use different ports; keep for completeness) ---
    "rtsp://{user}:{pwd}@{ip}:10080/ch{ch}_1.264",
    "rtsp://admin:@{ip}:10080/ch{ch}_1.264",
]

# Probe parameters
PROBE_SECONDS = 4
OPEN_TIMEOUT_SECONDS = 6
READ_TIMEOUT_SECONDS = 4
EARLY_STOP_CONSECUTIVE_FAILS = 3  # stop discovery after N consecutive no-signal channels (if we found at least one working)

# After the first channel connects, reuse that exact URL pattern for the rest.
LOCK_TEMPLATE_AFTER_FIRST_MATCH = True   # if True, prefer the first successful template for all channels
STRICT_LOCKED_TEMPLATE = True            # if True, do NOT fallback to other templates for later channels when locked
SELECTED_TEMPLATE: Optional[str] = None  # will hold the winning template string once discovered

# Force RTSP over TCP and reduce buffering (supported by OpenCV+FFmpeg)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000|max_delay;5000000|buffer_size;1048576|allowed_media_types;video")
# ---------------------------------------------------


def encode_creds(user: str, pwd: str) -> Tuple[str, str]:
    return quote(user, safe=""), quote(pwd, safe="")


def urls_from_template_for_channel(tmpl: str, ch: int) -> List[str]:
    """
    Produce one or more candidate URLs for this specific template and channel,
    trying both non-padded and zero-padded channel numbers where applicable.
    """
    u, p = encode_creds(NVR_USER, NVR_PASS)
    ch_variants = [f"{ch}", f"{ch:02d}"]
    hik_cc = f"{ch:02d}"
    urls = []
    for ch_str in ch_variants:
        try:
            url = tmpl.format(user=u, pwd=p, ip=NVR_IP, port=NVR_PORT, ch=ch_str, hik_cc=hik_cc)
        except KeyError:
            # template might not use all keys
            try:
                url = tmpl.format(user=u, pwd=p, ip=NVR_IP, port=NVR_PORT, ch=ch_str)
            except Exception:
                continue
        urls.append(url)
    # Dedupe preserve order
    out, seen = [], set()
    for u_ in urls:
        if u_ not in seen:
            seen.add(u_)
            out.append(u_)
    return out


def candidate_urls(ch: int) -> List[str]:
    u, p = encode_creds(NVR_USER, NVR_PASS)
    ch_variants = [f"{ch}", f"{ch:02d}"]  # e.g., "1" and "01"
    hik_cc = f"{ch:02d}"                  # Hik uses 2-digit channel prefix

    urls = []
    for t in TRY_TEMPLATES:
        for ch_str in ch_variants:
            try:
                url = t.format(user=u, pwd=p, ip=NVR_IP, port=NVR_PORT, ch=ch_str, hik_cc=hik_cc)
            except KeyError:
                # Template might not have all keys; skip if formatting fails
                continue
            urls.append(url)

    # Dedupe while preserving order
    seen = set()
    out = []
    for u_ in urls:
        if u_ not in seen:
            seen.add(u_)
            out.append(u_)
    return out


def probe_rtsp(url: str, seconds: int = PROBE_SECONDS) -> bool:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    # Try to hint timeouts if the backend honors them
    try:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_SECONDS * 1000)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_SECONDS * 1000)
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        return False

    # Warm-up: allow a little time for SPS/PPS and first keyframe
    t0 = time.time()
    first_ok = False
    while time.time() - t0 < 1.5:
        ok = cap.grab()
        if ok:
            first_ok = True
            break
        time.sleep(0.05)

    if not first_ok:
        cap.release()
        return False

    # Read frames for stability
    end = time.time() + max(1, seconds)
    good = True
    while time.time() < end:
        ok, _ = cap.read()
        if not ok:
            good = False
            break
    cap.release()
    return good


def find_working_url_for_channel(ch: int) -> Tuple[Optional[str], Optional[str]]:
    # If a template has been locked in, try it first (and maybe only)
    global SELECTED_TEMPLATE
    if LOCK_TEMPLATE_AFTER_FIRST_MATCH and SELECTED_TEMPLATE:
        for url in urls_from_template_for_channel(SELECTED_TEMPLATE, ch):
            if probe_rtsp(url, seconds=2):
                return url, SELECTED_TEMPLATE
        if STRICT_LOCKED_TEMPLATE:
            return None, SELECTED_TEMPLATE  # do not try other templates when strictly locked
        # else fall through to try other templates as fallback
    # No locked template yet, or fallback allowed
    for tmpl in TRY_TEMPLATES:
        for url in urls_from_template_for_channel(tmpl, ch):
            if probe_rtsp(url, seconds=2):
                return url, tmpl
    return None, None


class StreamReader(threading.Thread):
    def __init__(self, ch: int, url: str, frame_store: Dict[int, Tuple[np.ndarray, float]], lock: threading.Lock):
        super().__init__(daemon=True)
        self.ch = ch
        self.url = url
        self.frame_store = frame_store
        self.lock = lock
        self.stop_flag = threading.Event()

    def run(self):
        # Try to open with retries
        backoff = 1.0
        while not self.stop_flag.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
                continue

            # Optional: tune buffer size/latency (not all builds honor these)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            last_frame_time = time.time()
            while not self.stop_flag.is_set():
                ok, frame = cap.read()
                now = time.time()
                if not ok:
                    # if we stall for too long, reconnect
                    if now - last_frame_time > READ_TIMEOUT_SECONDS:
                        cap.release()
                        break
                    else:
                        # brief hiccup—sleep a tad and continue
                        time.sleep(0.02)
                        continue

                last_frame_time = now
                with self.lock:
                    self.frame_store[self.ch] = (frame, now)

            # reconnect loop will continue
        # end run

    def stop(self):
        self.stop_flag.set()


def draw_label(img: np.ndarray, text: str, color=(255, 255, 255)) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def make_placeholder(w: int, h: int, text: str) -> np.ndarray:
    tile = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(tile, text, (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    return tile


def main():
    assert GRID_ROWS * GRID_COLS >= CHANNEL_COUNT, "Grid is smaller than number of channels."

    print("Discovering working RTSP URLs per channel...")
    urls: Dict[int, Optional[str]] = {}
    consecutive_fail = 0
    last_ch = 0
    found_any = False
    for ch in range(1, CHANNEL_COUNT + 1):
        last_ch = ch
        url, tmpl = find_working_url_for_channel(ch)
        if url:
            print(f"[{ch}] ✓ {url}")
            consecutive_fail = 0
            found_any = True
            if LOCK_TEMPLATE_AFTER_FIRST_MATCH and tmpl and SELECTED_TEMPLATE is None:
                SELECTED_TEMPLATE = tmpl
                print(f"[lock] Using template for all channels: {SELECTED_TEMPLATE}")
        else:
            # If we are strictly locked to a template and it failed, mark fail
            lock_note = " (locked template)" if (LOCK_TEMPLATE_AFTER_FIRST_MATCH and SELECTED_TEMPLATE and STRICT_LOCKED_TEMPLATE) else ""
            print(f"[{ch}] ✗ No working URL found{lock_note}")
            consecutive_fail += 1
        urls[ch] = url
        if found_any and consecutive_fail >= EARLY_STOP_CONSECUTIVE_FAILS:
            print(f"Early stop: {consecutive_fail} consecutive no-signal channels after CH {ch}. Proceeding with discovered channels 1..{last_ch}.")
            break
    # we will only render channels that were actually probed (1..last_ch)
    total_channels = last_ch

    # shared frame store (latest frame, timestamp) per channel
    frame_store: Dict[int, Tuple[np.ndarray, float]] = {}
    lock = threading.Lock()

    # Start readers
    readers: List[StreamReader] = []
    for ch, url in urls.items():
        if url is None:
            continue
        r = StreamReader(ch, url, frame_store, lock)
        r.start()
        readers.append(r)

    cell_w = CANVAS_W // GRID_COLS
    cell_h = CANVAS_H // GRID_ROWS

    # Pre-create placeholders
    placeholders = {
        ch: make_placeholder(cell_w, cell_h, f"CH {ch} - NO SIGNAL")
        for ch in range(1, total_channels + 1)
    }

    # Mosaic loop
    delay = max(1, int(1000 / max(1, TARGET_FPS)))  # in ms for waitKey
    window_name = f"NVR Mosaic {GRID_COLS}x{GRID_ROWS} ({total_channels} ch)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CANVAS_W, CANVAS_H)

    try:
        while True:
            tiles = []
            with lock:
                for ch in range(1, total_channels + 1):
                    if ch in frame_store:
                        frame, ts = frame_store[ch]
                        # resize to cell
                        if frame is not None:
                            fr = cv2.resize(frame, (cell_w, cell_h))
                            age = time.time() - ts
                            label = f"CH {ch}  {'LIVE' if age < 2.0 else f'LAG {age:.1f}s'}"
                            draw_label(fr, label, (0, 255, 0) if age < 2.0 else (0, 200, 255))
                            tiles.append(fr)
                        else:
                            tiles.append(placeholders[ch].copy())
                    else:
                        tiles.append(placeholders[ch].copy())

            # pad tiles list to full grid (if total_channels < GRID_ROWS*GRID_COLS)
            total_cells = GRID_ROWS * GRID_COLS
            if len(tiles) < total_cells:
                tiles += [np.zeros((cell_h, cell_w, 3), dtype=np.uint8)] * (total_cells - len(tiles))

            # assemble mosaic
            rows = []
            for r in range(GRID_ROWS):
                row_tiles = tiles[r * GRID_COLS:(r + 1) * GRID_COLS]
                rows.append(np.hstack(row_tiles))
            mosaic = np.vstack(rows)

            cv2.imshow(window_name, mosaic)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
    finally:
        for r in readers:
            r.stop()
        # give threads a moment to exit
        time.sleep(0.5)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()