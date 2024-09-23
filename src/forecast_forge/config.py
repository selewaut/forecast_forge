from pathlib import Path

# Define the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
#  Define paths
DATA_DIR = PROJECT_ROOT / "data" / "walmart_sales_forecasting"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
STORES_PATH = DATA_DIR / "stores.csv"
