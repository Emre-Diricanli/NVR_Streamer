#!/usr/bin/env python3
"""
RTSP hardware-accelerated viewer (self-contained)

Just edit the RTSP_URL variable below and run:
    python rtsp_hw_fixed.py
"""

import cv2
import os
import time
import platform
import sys


# === EDIT THIS LINE ===
RTSP_URL = "rtsp://admin:pvs12385@192.168.1.168:554/Streaming/Channels/1701"
# ======================

# Optional config
HW_BACKEND = "auto"       # "auto", "nvdec", "vaapi", "vtb", "cpu"
CODEC = "auto"            # "auto", "h264", "h265"
TRANSPORT = "tcp"         # "tcp" or "udp"
LATENCY_MS = 100
WINDOW_TITLE = "RTSP Stream"
RECONNECT_SECONDS = 3.0


def guess_platform_hw():
    sysname = platform.system().lower()
    if sysname == "darwin":
        return "vtb"
    if sysname == "linux":
        if os.path.isdir("/proc/driver/nvidia") or os.path.exists("/dev/nvidiactl"):
            return "nvdec"
        return "vaapi"
    return "cpu"


def make_gst_pipeline(url: str,
                      hw: str = "auto",
                      codec: str = "auto",
                      transport: str = "tcp",
                      latency_ms: int = 100) -> str:
    if hw == "auto":
        hw = guess_platform_hw()

    trans_flag = "GST_RTSP_LOWER_TRANS_TCP" if transport.lower() == "tcp" else "GST_RTSP_LOWER_TRANS_UDP"
    dec_map = {
        ("nvdec", "h264"): "nvh264dec",
        ("nvdec", "h265"): "nvh265dec",
        ("vaapi", "h264"): "vaapih264dec",
        ("vaapi", "h265"): "vaapih265dec",
        ("vtb", "h264"): "vtdec",
        ("vtb", "h265"): "vtdec",
        ("cpu", "h264"): "avdec_h264",
        ("cpu", "h265"): "avdec_h265",
    }

    def pick_decoder(hw_choice, codec_choice, caps_codec):
        if codec_choice in ("h264", "h265"):
            return dec_map.get((hw_choice, codec_choice))
        return dec_map.get((hw_choice, caps_codec), None)

    h264_dec = pick_decoder(hw, codec, "h264") or "decodebin"
    h265_dec = pick_decoder(hw, codec, "h265") or "decodebin"

    base = f'rtspsrc location="{url}" latency={latency_ms} protocols={trans_flag} drop-on-latency=true ! '
    h264 = f"rtph264depay ! h264parse config-interval=1 ! {h264_dec} ! videoconvert qos=false ! video/x-raw,format=BGR ! appsink max-buffers=2 drop=true sync=false"
    h265 = f"rtph265depay ! h265parse config-interval=1 ! {h265_dec} ! videoconvert qos=false ! video/x-raw,format=BGR ! appsink max-buffers=2 drop=true sync=false"
    pipeline = base + f"tee name=t t. ! queue leaky=downstream ! {h264} t. ! queue leaky=downstream ! {h265}"
    return pipeline


def open_with_gstreamer(pipeline: str):
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return cap if cap.isOpened() else None


def open_with_ffmpeg(url: str):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    return cap if cap.isOpened() else None


def main():
    pipeline = make_gst_pipeline(RTSP_URL, HW_BACKEND, CODEC, TRANSPORT, LATENCY_MS)
    last_backend = ""

    while True:
        cap = open_with_gstreamer(pipeline) or open_with_ffmpeg(RTSP_URL)
        if cap is None:
            print("ERROR: failed to open RTSP stream.")
            sys.exit(2)

        backend = "GStreamer" if last_backend != "FFmpeg" else "FFmpeg"
        print(f"[info] opened {RTSP_URL} using {backend} backend. Press 'q' to quit.")
        fps_ts, frames = time.time(), 0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[warn] read failed, reconnecting...")
                break

            frames += 1
            now = time.time()
            if now - fps_ts >= 1.0:
                print(f"[fps] {frames}")
                frames, fps_ts = 0, now

            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()
        time.sleep(RECONNECT_SECONDS)


if __name__ == "__main__":
    main()