from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
# Change this to the folder that directly contains your 107 pose-class folders.
DATASET_DIR = Path(r"C:\Users\HP\Yoga-posture-assessment\dataset")
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RANDOM_STATE = 42
TEST_SIZE = 0.20
