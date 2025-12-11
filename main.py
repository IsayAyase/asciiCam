import cv2
import numpy as np
from PIL import Image
import zstandard as zstd
import struct
from tqdm import tqdm
import argparse
import os

DENSITY_STRING = [' ', '.', ',', '-', '=', '+', ':', ';', 'c', 'b', 'a',
                  '!', '?', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '$', 'W', '#', '@', 'N']
DENSITY_LEN = len(DENSITY_STRING)

def get_char_index_from_brightness(brightness: np.ndarray):
    """Map pixel brightness (0–255) to density index."""
    indices = np.clip((brightness / (256 / DENSITY_LEN)).astype(int), 0, DENSITY_LEN - 1)
    return indices


def process_frame(frame: np.ndarray, width: int, height: int):
    """Resize frame and compute ASCII char indices + avg colors."""
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    # Brightness from RGB average
    brightness = resized.mean(axis=2)
    char_indices = get_char_index_from_brightness(brightness).astype(np.uint8)

    # Flatten and compute avg color (each pixel already 1:1 with char)
    colors = resized.reshape(-1, 3).astype(np.uint8)
    return char_indices.flatten(), colors


def write_header(f, width, height, fps, total_frames):
    """Write binary header for the .bin file"""
    f.write(b'ASCI')  # Magic bytes
    f.write(struct.pack("HHH", width, height, fps))
    f.write(struct.pack("I", total_frames))


def generate_binary(video_path: str, width: int, height: int, fps: int, output_path: str):
    cap = cv2.VideoCapture(video_path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = max(int(actual_fps / fps), 1)
    print(f"Input FPS: {actual_fps:.2f}, sampling every {frame_interval} frame(s)")

    tmp_file = output_path + ".raw"

    with open(tmp_file, "wb") as f:
        # We'll count actual processed frames
        processed = 0
        header_placeholder = b'ASCI' + struct.pack("<HHHI", width, height, fps, 0)
        f.write(header_placeholder)

        frame_id = 0
        pbar = tqdm(total=total_frames, desc="Processing frames")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                char_indices, colors = process_frame(frame, width, height)
                combined = np.column_stack((char_indices, colors)).astype(np.uint8)
                f.write(combined.tobytes())
                processed += 1

            frame_id += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        # Now go back and patch total_frames
        f.seek(4 + 2 + 2 + 2)  # skip magic + width + height + fps
        f.write(struct.pack("<I", processed))

    # Compress with Zstandard
    # print(f"Compressing {processed} frames...")
    # with open(tmp_file, "rb") as f_in, open(output_path, "wb") as f_out:
    #     compressor = zstd.ZstdCompressor(level=10)
    #     compressor.copy_stream(f_in, f_out)

    # os.remove(tmp_file)

    os.rename(tmp_file, output_path)
    print(f"✅ Done: {output_path} ({processed} frames written)")


# ------------------- CLI ENTRY -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess video → compressed binary ASCII frames")
    parser.add_argument("--v", type=str, required=True, help="Path to input video file")
    parser.add_argument("--w", type=int, default=120, help="Target character width")
    parser.add_argument("--h", type=int, default=60, help="Target character height")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (sampling rate)")
    parser.add_argument("--o", type=str, default="ascii_video.bin", help="Output binary file path")

    args = parser.parse_args()
    generate_binary(args.v, args.w, args.h, args.fps, args.o)

    # uv run main.py --v /d/ascii-canvas-frame-generator/generator/test-videos/ice.mp4 --w 128 --h 72 --fps 30 --o ascii_video.bin