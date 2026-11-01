"""
Smart Camera - Laptop Webcam Streamer for Google Colab
-------------------------------------------------------
Streams your local laptop webcam (or video file) live to your Google Colab instance,
where Colab's cloud GPU (NVIDIA T4) runs real-time YOLO object detection, posture
tracking (sitting vs standing), dwell timing, and dashboard analytics.

Usage:
    python send_webcam_to_colab.py --url https://your-tunnel-url.trycloudflare.com
    python send_webcam_to_colab.py --url http://127.0.0.1:8000 --source 0
"""

import argparse
import sys
import time
import cv2
import requests


def main():
    parser = argparse.ArgumentParser(description="Stream laptop webcam to Google Colab Smart Camera")
    parser.add_argument("--url", type=str, required=True, help="Colab Cloudflare Tunnel URL (e.g. https://xyz.trycloudflare.com)")
    parser.add_argument("--source", default=0, help="Webcam device index (0) or path to a local video file")
    parser.add_argument("--fps", type=int, default=15, help="Target transmission frame rate (default: 15 FPS)")
    parser.add_argument("--quality", type=int, default=70, help="JPEG compression quality (1-100, default: 70)")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    upload_url = f"{base_url}/api/camera/upload-raw-frame"

    # Parse source as int if numeric
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    print("=" * 65)
    print(" SMART CAMERA - LIVE STREAMER TO GOOGLE COLAB")
    print("=" * 65)
    print(f" Target URL     : {upload_url}")
    print(f" Camera Source  : {source}")
    print(f" Target FPS     : {args.fps}")
    print(f" Quality        : {args.quality}% JPEG")
    print(" Press Ctrl+C in this terminal to stop streaming.")
    print("=" * 65)

    # Initialize video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Error] Could not open camera source {source}. Make sure no other app is using your webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_delay = 1.0 / max(1, args.fps)
    session = requests.Session()
    session.headers.update({"Content-Type": "image/jpeg"})

    frame_count = 0
    start_time = time.time()
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[Warning] Frame grab failed or video reached the end.")
                break

            # Resize if frame is larger than 640x480 to preserve bandwidth
            h, w = frame.shape[:2]
            if w > 640 or h > 480:
                frame = cv2.resize(frame, (640, 480))

            # Encode to JPEG
            ok, buffer = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue

            # Send to Colab
            try:
                resp = session.post(upload_url, data=buffer.tobytes(), timeout=0.8)
                if resp.status_code == 200:
                    frame_count += 1
                else:
                    print(f"\n[Warning] Server returned status {resp.status_code}")
            except requests.exceptions.RequestException as e:
                # Glitch or transient network delay
                pass

            # Print status every 30 frames
            if frame_count > 0 and frame_count % 30 == 0:
                elapsed = time.time() - start_time
                curr_fps = frame_count / max(0.1, elapsed)
                sys.stdout.write(f"\r[Streaming] Active | Frames Sent: {frame_count} | Effective Rate: {curr_fps:.1f} FPS   ")
                sys.stdout.flush()

            # Rate limiter
            spent = time.time() - t0
            sleep_time = max(0.001, frame_delay - spent)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n[Stopped] Stream stopped by user.")
    finally:
        cap.release()
        session.close()
        print("Camera released. Exiting.")


if __name__ == "__main__":
    main()
