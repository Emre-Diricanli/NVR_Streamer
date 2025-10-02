#!/usr/bin/env python3
# nvr_grid_mosaic_fast.py

import os, time, threading
from urllib.parse import quote
from typing import Dict, Optional, Tuple, List
import cv2, numpy as np

# ----------------- CONFIG -----------------
NVR_IP = "70.168.44.34"
NVR_PORT = 1024Zu
NVR_USER = "admin"
NVRPASS = "Zuppardos1"

CHANNEL_COUNT = 20
GRID_COLS, GRID_ROWS = 4, 5          # 4x3 grid for 12 channels
CANVAS_W, CANVAS_H = 1440, 810       # slightly smaller → faster
TARGET_FPS = 12                      # cap mosaic refresh rate

# Prefer substreams FIRST (lighter). Keep a few mains as fallback.
TRY_TEMPLATES = [
    # Substreams first (lower resolution/bitrate)
    # "rtsp://{user}:{pwd}@{ip}:{port}/Streaming/Channels/{ch}02",   # Hik sub
    # "rtsp://{user}:{pwd}@{ip}:{port}/cam/realmonitor?channel={ch}&subtype=1",  # Dahua sub

    # Then mains if subs not found
    "rtsp://{user}:{pwd}@{ip}:{port}/Streaming/Channels/{ch}01",   # Hik main
    "rtsp://{user}:{pwd}@{ip}:{port}/cam/realmonitor?channel={ch}&subtype=0",  # Dahua main

    # Others
    "rtsp://{user}:{pwd}@{ip}:{port}/h264/ch{ch}/main/av_stream",
    "rtsp://{user}:{pwd}@{ip}:{port}/ch{ch}/0",
    "rtsp://{user}:{pwd}@{ip}:{port}/live/ch{ch}",
    "rtsp://{user}:{pwd}@{ip}:{port}/media/video{ch}",
]

# Probe/timeout
PROBE_SECONDS = 2
READ_TIMEOUT_SECONDS = 3

# Low-latency ffmpeg options (tweak if unstable)
# - tcp transport, tiny buffers, no reordering, low delay
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|probesize;8192|"
    "analyzeduration;0|max_delay;250000|reorder_queue_size;0|stimeout;5000000"
)
# ------------------------------------------


def enc(s: str) -> str:
    return quote(s, safe="")

def candidates(ch: int) -> List[str]:
    u, p = enc(NVR_USER), enc(NVR_PASS)
    ch1, ch2 = f"{ch}", f"{ch:02d}"
    urls, seen = [], set()
    for t in TRY_TEMPLATES:
        for c in (ch1, ch2):  # try non-padded and zero-padded
            url = t.format(user=u, pwd=p, ip=NVR_IP, port=NVR_PORT, ch=c)
            if url not in seen:
                urls.append(url); seen.add(url)
    return urls

def probe(url: str, seconds: int = PROBE_SECONDS) -> bool:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return False
    ok, _ = cap.read()
    if not ok:
        cap.release(); return False
    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, _ = cap.read()
        if not ok:
            cap.release(); return False
    cap.release()
    return True

def find_url(ch: int) -> Optional[str]:
    for url in candidates(ch):
        if probe(url, 1):
            return url
    return None

class Reader(threading.Thread):
    def __init__(self, ch: int, url: str, store: Dict[int, Tuple[np.ndarray, float]], lock: threading.Lock):
        super().__init__(daemon=True)
        self.ch, self.url, self.store, self.lock = ch, url, store, lock
        self.stop_flag = threading.Event()

    def run(self):
        backoff = 0.5
        while not self.stop_flag.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                time.sleep(backoff); backoff = min(backoff * 2, 4.0); continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            last = time.time()
            while not self.stop_flag.is_set():
                ok, frame = cap.read()
                now = time.time()
                if not ok:
                    if now - last > READ_TIMEOUT_SECONDS:
                        cap.release(); break
                    time.sleep(0.01); continue
                last = now
                # store latest only
                with self.lock:
                    self.store[self.ch] = (frame, now)

    def stop(self): self.stop_flag.set()

def draw_label_inplace(tile: np.ndarray, label: str, fresh: bool):
    # Minimal overlay: a slim bar + text; keep it cheap.
    h, w = tile.shape[:2]
    cv2.rectangle(tile, (0,0), (w, 22), (0,0,0), -1)
    color = (0,255,0) if fresh else (0,200,255)
    cv2.putText(tile, label, (6,16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def main():
    assert GRID_COLS * GRID_ROWS >= CHANNEL_COUNT, "Grid too small."

    print("Discovering streams (substreams prioritized)...")
    urls: Dict[int, Optional[str]] = {}
    for ch in range(1, CHANNEL_COUNT + 1):
        u = find_url(ch)
        print(f"[{ch}] {'✓ ' + u if u else '✗ no stream'}")
        urls[ch] = u

    # Start readers
    store: Dict[int, Tuple[np.ndarray, float]] = {}
    lock = threading.Lock()
    readers: List[Reader] = []
    for ch, u in urls.items():
        if u:
            r = Reader(ch, u, store, lock)
            r.start(); readers.append(r)

    # Precompute geometry and allocate one big mosaic buffer once
    cell_w, cell_h = CANVAS_W // GRID_COLS, CANVAS_H // GRID_ROWS
    mosaic = np.zeros((cell_h * GRID_ROWS, cell_w * GRID_COLS, 3), dtype=np.uint8)

    # Precompute tile slices to avoid per-frame slicing math
    slices: List[Tuple[slice, slice]] = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            rs = slice(r*cell_h, (r+1)*cell_h)
            cs = slice(c*cell_w, (c+1)*cell_w)
            slices.append((rs, cs))

    # Placeholders per channel
    placeholders = {ch: np.zeros((cell_h, cell_w, 3), dtype=np.uint8) for ch in range(1, CHANNEL_COUNT+1)}
    for ch in placeholders:
        cv2.putText(placeholders[ch], f"CH {ch} - NO SIGNAL", (10, cell_h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2, cv2.LINE_AA)

    window = f"NVR Mosaic FAST {GRID_COLS}x{GRID_ROWS}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, CANVAS_W, CANVAS_H)

    delay_ms = max(1, int(1000 / max(1, TARGET_FPS)))
    try:
        while True:
            # Fill mosaic in place → no np.hstack/vstack allocations
            idx = 0
            now = time.time()
            with lock:
                for ch in range(1, CHANNEL_COUNT + 1):
                    rs, cs = slices[idx]; idx += 1
                    if ch in store:
                        frame, ts = store[ch]
                        # fast resize; INTER_AREA is efficient for downscale
                        if frame is not None:
                            fr = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                            mosaic[rs, cs] = fr
                            draw_label_inplace(mosaic[rs, cs], f"CH {ch}", fresh=(now - ts < 1.5))
                        else:
                            mosaic[rs, cs] = placeholders[ch]
                    else:
                        mosaic[rs, cs] = placeholders[ch]

            cv2.imshow(window, mosaic)
            if (cv2.waitKey(delay_ms) & 0xFF) == ord('q'):
                break
    finally:
        for r in readers: r.stop()
        time.sleep(0.3)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
