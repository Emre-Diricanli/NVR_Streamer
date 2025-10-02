#!/usr/bin/env python3
"""
Show all NVR channels in ONE tiled window (grid mosaic) with optional GPU decode.

- Discovers a working RTSP URL per channel using common templates.
- Starts one reader thread per channel (stores latest frame only).
- Renders a live grid window with overlays.
- Prefers GStreamer + HW decode (NVDEC / VA-API). Falls back to FFmpeg/CPU.

Quit: press 'q' while the mosaic window is focused.

Requirements:
  - Python 3.8+
  - pip install opencv-python
  - System: GStreamer runtime (see notes above) for GPU path
"""

import os
import time
import threading
from urllib.parse import quote
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np

# ----------------- CONFIGURE THESE -----------------
NVR_IP = "192.168.0.71"
NVR_PORT = 554     
NVR_USER = "ai"
NVR_PASS = "1880teso@"

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
    # "rtsp://{user}:{pwd}@{ip}:{port}/Streaming/Channels/{hik_cc}01",
    "rtsp://{user}:{pwd}@{ip}:{port}/Streaming/Channels/{hik_cc}02",

    # --- Dahua / Amcrest ---
    # "rtsp://{user}:{pwd}@{ip}:{port}/cam/realmonitor?channel={ch}&subtype=0",
    "rtsp://{user}:{pwd}@{ip}:{port}/cam/realmonitor?channel={ch}&subtype=1",

    # --- Web Client ---
    "rtsp://{user}:{pwd}@{ip}:{port}/unicast/c{ch}/s1/live",
    "rtsp://{user}:{pwd}@{ip}:{port}/unicast/c{ch}/s0/live",

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

# Force RTSP over TCP, reduce buffering (FFmpeg backend). GStreamer path ignores this.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|max_delay;5000000|buffer_size;1048576|allowed_media_types;video"
)

# Choose GPU vendor for GStreamer path: nvidia | intel | amd | vaapi | auto
GST_GPU_VENDOR = os.getenv("GST_GPU_VENDOR", "auto").lower()

# ---------------------------------------------------
#  GPU/GStreamer Helpers
# ---------------------------------------------------

def have_gstreamer() -> bool:
    """
    Quick check: CAP_GSTREAMER constant exists and backend is available.
    """
    try:
        _ = cv2.CAP_GSTREAMER
        return True
    except AttributeError:
        return False


def _gs_common_front(rtsp_url: str) -> str:
    # rtspsrc protocols=tcp to reduce packet loss; latency tweak for smoother playback
    return (f'rtspsrc location="{rtsp_url}" protocols=tcp latency=200 ! '
            'rtpjitterbuffer ! ')


def _build_hw_pipelines(rtsp_url: str, vendor: str = "auto") -> List[str]:
    """
    Build a list of candidate GStreamer pipelines that attempt HW decode first.
    We try both H.264 and H.265 variants, with multiple decoder element fallbacks.
    """
    front_h264 = _gs_common_front(rtsp_url) + 'rtph264depay ! h264parse ! '
    front_h265 = _gs_common_front(rtsp_url) + 'rtph265depay ! h265parse ! '

    # Common tail: convert to BGR for OpenCV in system memory
    tail = 'videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false'

    pipelines: List[str] = []

    vend = vendor
    if vend == "intel" or vend == "amd":
        vend = "vaapi"
    if vend not in ("nvidia", "vaapi", "auto"):
        vend = "auto"

    if vend in ("nvidia", "auto"):
        # NVIDIA NVDEC (requires gstreamer NVDEC plugins + driver)
        # H.264 / H.265 decoders:
        pipelines += [
            front_h264 + 'nvh264dec ! ' + tail,
            front_h265 + 'nvh265dec ! ' + tail,
        ]

    if vend in ("vaapi", "auto"):
        # VA-API (Intel/AMD). Newer GStreamer has vah264dec/vah265dec; older uses vaapih264dec/vaapih265dec.
        pipelines += [
            front_h264 + 'vah264dec ! ' + tail,        # new
            front_h265 + 'vah265dec ! ' + tail,        # new
            front_h264 + 'vaapih264dec ! ' + tail,     # legacy fallback
            front_h265 + 'vaapih265dec ! ' + tail,     # legacy fallback
        ]

    # As the last resort on the GStreamer path: decodebin (may still pick HW if available)
    pipelines += [
        _gs_common_front(rtsp_url) + 'decodebin ! ' + tail
    ]
    return pipelines


def _open_with_gst(rtsp_url: str) -> Optional[cv2.VideoCapture]:
    """
    Try multiple GStreamer pipelines to open the RTSP stream with HW accel if possible.
    Returns an opened VideoCapture or None.
    """
    if not have_gstreamer():
        return None

    candidates = _build_hw_pipelines(rtsp_url, vendor=GST_GPU_VENDOR)
    for pipe in candidates:
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def _open_with_ffmpeg(rtsp_url: str) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_SECONDS * 1000)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_SECONDS * 1000)
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def open_capture(rtsp_url: str) -> Optional[cv2.VideoCapture]:
    """
    Prefer GStreamer (HW) path. Fall back to FFmpeg (CPU) path.
    """
    cap = _open_with_gst(rtsp_url)
    if cap is not None:
        return cap
    return _open_with_ffmpeg(rtsp_url)


# ---------------------------------------------------
#  Original helpers (slightly adapted)
# ---------------------------------------------------

def encode_creds(user: str, pwd: str) -> Tuple[str, str]:
    return quote(user, safe=""), quote(pwd, safe="")


def urls_from_template_for_channel(tmpl: str, ch: int) -> List[str]:
    u, p = encode_creds(NVR_USER, NVR_PASS)
    ch_variants = [f"{ch}", f"{ch:02d}"]
    hik_cc = f"{ch:02d}"
    urls = []
    for ch_str in ch_variants:
        try:
            url = tmpl.format(user=u, pwd=p, ip=NVR_IP, port=NVR_PORT, ch=ch_str, hik_cc=hik_cc)
        except KeyError:
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
    ch_variants = [f"{ch}", f"{ch:02d}"]
    hik_cc = f"{ch:02d}"

    urls = []
    for t in TRY_TEMPLATES:
        for ch_str in ch_variants:
            try:
                url = t.format(user=u, pwd=p, ip=NVR_IP, port=NVR_PORT, ch=ch_str, hik_cc=hik_cc)
            except KeyError:
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
    """
    Try to open using GStreamer (HW) first, then FFmpeg (CPU) as fallback.
    Read a couple seconds to verify stability.
    """
    cap = open_capture(url)
    if cap is None or not cap.isOpened():
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
    global SELECTED_TEMPLATE
    if LOCK_TEMPLATE_AFTER_FIRST_MATCH and SELECTED_TEMPLATE:
        for url in urls_from_template_for_channel(SELECTED_TEMPLATE, ch):
            if probe_rtsp(url, seconds=2):
                return url, SELECTED_TEMPLATE
        if STRICT_LOCKED_TEMPLATE:
            return None, SELECTED_TEMPLATE
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
        backoff = 1.0
        while not self.stop_flag.is_set():
            cap = open_capture(self.url)
            if cap is None or not cap.isOpened():
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
                continue

            # optional: try to keep latency low; not all backends honor these
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            last_frame_time = time.time()
            while not self.stop_flag.is_set():
                ok, frame = cap.read()
                now = time.time()
                if not ok:
                    if now - last_frame_time > READ_TIMEOUT_SECONDS:
                        cap.release()
                        break
                    else:
                        time.sleep(0.02)
                        continue

                last_frame_time = now
                with self.lock:
                    self.frame_store[self.ch] = (frame, now)
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
    global SELECTED_TEMPLATE
    assert GRID_ROWS * GRID_COLS >= CHANNEL_COUNT, "Grid is smaller than number of channels."

    print(f"GPU vendor (GStreamer): {GST_GPU_VENDOR}")
    if have_gstreamer():
        print("GStreamer backend detected. Will try HW decode first.")
    else:
        print("GStreamer backend not detected in OpenCV. Using FFmpeg/CPU path.")

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
            lock_note = " (locked template)" if (LOCK_TEMPLATE_AFTER_FIRST_MATCH and SELECTED_TEMPLATE and STRICT_LOCKED_TEMPLATE) else ""
            print(f"[{ch}] ✗ No working URL found{lock_note}")
            consecutive_fail += 1
        urls[ch] = url
        if found_any and consecutive_fail >= EARLY_STOP_CONSECUTIVE_FAILS:
            print(f"Early stop: {consecutive_fail} consecutive no-signal channels after CH {ch}. Proceeding with discovered channels 1..{last_ch}.")
            break

    total_channels = last_ch

    frame_store: Dict[int, Tuple[np.ndarray, float]] = {}
    lock = threading.Lock()

    readers: List[StreamReader] = []
    for ch, url in urls.items():
        if url is None:
            continue
        r = StreamReader(ch, url, frame_store, lock)
        r.start()
        readers.append(r)

    cell_w = CANVAS_W // GRID_COLS
    cell_h = CANVAS_H // GRID_ROWS

    placeholders = {
        ch: make_placeholder(cell_w, cell_h, f"CH {ch} - NO SIGNAL")
        for ch in range(1, total_channels + 1)
    }

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

            total_cells = GRID_ROWS * GRID_COLS
            if len(tiles) < total_cells:
                tiles += [np.zeros((cell_h, cell_w, 3), dtype=np.uint8)] * (total_cells - len(tiles))

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
        time.sleep(0.5)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()