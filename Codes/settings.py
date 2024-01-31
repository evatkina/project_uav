from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
VIDEO = 'Video'
SOURCES_LIST = [VIDEO]

# Videos config
VIDEO_DIR = ROOT / 'testvideo' / 'Obj_Video'
VIDEO_1_PATH = VIDEO_DIR / 'cameraVideo.avi'
VIDEO_2_PATH = VIDEO_DIR / 'ThermalVideoNear.mp4'
VIDEO_3_PATH = VIDEO_DIR / 'CameraVideoFar.mp4'
VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH,
    'video_2': VIDEO_2_PATH,
    'video_3': VIDEO_3_PATH,
}

# ML Model config
