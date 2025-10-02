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
    # Hikvision (main/sub)
    # "rtsp://{user}:{pwd}@{ip}:{port}/Streaming/Channels/{ch}01",
     "rtsp://{user}:{pwd}@{ip}:{port}/Streaming/Channels/{ch}02",

    # Dahua/Amcrest
    # "rtsp://{user}:{pwd}@{ip}:{port}/cam/realmonitor?channel={ch}&subtype=0",
    "rtsp://{user}:{pwd}@{ip}:{port}/cam/realmonitor?channel={ch}&subtype=1",

    # Others
    "rtsp://{user}:{pwd}@{ip}:{port}/h264/ch{ch}/main/av_stream",
    "rtsp://{user}:{pwd}@{ip}:{port}/ch{ch}/0",
    "rtsp://{user}:{pwd}@{ip}:{port}/live/ch{ch}",
    "rtsp://{user}:{pwd}@{ip}:{port}/media/video{ch}",
    "rtsp://{user}:{pwd}@{ip}:{port}/Preview_{ch}_sub" #ReoLink
    "rtsp://{user}:{pwd}@{ip}:{port}/h264Preview_{ch}_sub" #ReoLink
    "rtsp://{user}:{pwd}@{ip}:{port}/Preview_{ch}_main" #ReoLink
    "rtsp://{user}:{pwd}@{ip}:{port}/h264Preview_{ch}_sub" #ReoLink

    "rtsp://admin:@{ip}:10080/ch{ch}_1.264" #Night-Owl Protect
]

# Probe parameters
PROBE_SECONDS = 4
OPEN_TIMEOUT_SECONDS = 6
READ_TIMEOUT_SECONDS = 4

# Force RTSP over TCP and reduce buffering (supported by OpenCV+FFmpeg)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|max_delay;5000000")
# ---------------------------------------------------


def encode_creds(user: str, pwd: str) -> Tuple[str, str]:
    return quote(user, safe=""), quote(pwd, safe="")


def candidate_urls(ch: int) -> List[str]:
    u, p = encode_creds(NVR_USER, NVR_PASS)
    ch1 = f"{ch}"
    ch2 = f"{ch:02d}"
    urls = []
    for t in TRY_TEMPLATES:
        urls.append(t.format(user=u, pwd=p, ip=NVR_IP, port=NVR_PORT, ch=ch1))
        u2 = t.format(user=u, pwd=p, ip=NVR_IP, port=NVR_PORT, ch=ch2)
        if u2 not in urls:
            urls.append(u2)
    # dedupe preserve order
    seen = set()
    out = []
    for u_ in urls:
        if u_ not in seen:
            seen.add(u_)
            out.append(u_)
    return out


def probe_rtsp(url: str, seconds: int = PROBE_SECONDS) -> bool:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return False
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return False
    # read a few frames for stability
    t0 = time.time()
    good = True
    while time.time() - t0 < seconds:
        ok, _ = cap.read()
        if not ok:
            good = False
            break
    cap.release()
    return good


def find_working_url_for_channel(ch: int) -> Optional[str]:
    for url in candidate_urls(ch):
        if probe_rtsp(url, seconds=2):
            return url
    return None


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
    for ch in range(1, CHANNEL_COUNT + 1):
        url = find_working_url_for_channel(ch)
        if url:
            print(f"[{ch}] ✓ {url}")
        else:
            print(f"[{ch}] ✗ No working URL found")
        urls[ch] = url

    # shared frame store (latest frame, timestamp) per channel
    frame_store: Dict[int, Tuple[np.ndarray, float]] = {}
    lock = threading.Lock()

    # Start readers
    readers: List[StreamReader] = []
    for ch, url in urls.items():
        if url:
            r = StreamReader(ch, url, frame_store, lock)
            r.start()
            readers.append(r)

    cell_w = CANVAS_W // GRID_COLS
    cell_h = CANVAS_H // GRID_ROWS

    # Pre-create placeholders
    placeholders = {
        ch: make_placeholder(cell_w, cell_h, f"CH {ch} - NO SIGNAL")
        for ch in range(1, CHANNEL_COUNT + 1)
    }

    # Mosaic loop
    delay = max(1, int(1000 / max(1, TARGET_FPS)))  # in ms for waitKey
    window_name = f"NVR Mosaic {GRID_COLS}x{GRID_ROWS} ({CHANNEL_COUNT} ch)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CANVAS_W, CANVAS_H)

    try:
        while True:
            tiles = []
            with lock:
                for ch in range(1, CHANNEL_COUNT + 1):
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

            # pad tiles list to full grid (if CHANNEL_COUNT < GRID_ROWS*GRID_COLS)
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