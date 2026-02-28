"""
Download MediaPipe Tasks .task models and run a quick extraction test.

This script downloads pose/hand/face .task models into
`sign_recognition/mediapipe_models/`, extracts the provided dataset zip to
`data/temp_sign`, finds the first video and runs the extractor to print
the feature shape.

Usage:
  python scripts/download_and_test_mediapipe_models.py --dataset-zip "C:/.../Home_4of4.zip"
"""

import argparse
import os
import urllib.request
import sys
import tempfile

# ensure project root is on sys.path when running from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def extract_zip(zip_path: str, dest_dir: str) -> str:
    import zipfile
    dest = os.path.abspath(dest_dir)
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest)
    return dest


MODEL_URLS = {
    'pose_landmarker.task': 'https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task',
    'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task',
    'face_landmarker.task': 'https://storage.googleapis.com/mediapipe-assets/face_landmarker.task',
}


def download_models(dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    for name, url in MODEL_URLS.items():
        out_path = os.path.join(dest_dir, name)
        if os.path.exists(out_path):
            print(f"Skipping existing {name}")
            continue
        print(f"Downloading {name} from {url}...")
        try:
            urllib.request.urlretrieve(url, out_path)
            print('Saved to', out_path)
        except Exception as e:
            print('Failed to download', url, e)
            raise


def find_first_video(root_dir: str):
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                return os.path.join(root, f)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-zip', required=True)
    p.add_argument('--model-dir', default=os.path.join('sign_recognition', 'mediapipe_models'))
    p.add_argument('--data-root', default='data/temp_sign')
    args = p.parse_args()

    download_models(args.model_dir)

    print('Extracting dataset...')
    extract_zip(args.dataset_zip, args.data_root)

    video = find_first_video(args.data_root)
    if not video:
        print('No video found in extracted dataset')
        sys.exit(1)

    print('Found video:', video)

    # Run a quick extraction using the project's extractor
    from sign_recognition.models.mediapipe_extractor import MediaPipeExtractor
    ext = MediaPipeExtractor(model_dir=args.model_dir)
    feats = ext.extract_video(video, max_frames=32)
    print('Extracted features shape:', feats.shape)
    ext.close()


if __name__ == '__main__':
    main()
