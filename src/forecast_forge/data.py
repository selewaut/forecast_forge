import kaggle
import zipfile
import os
from forecast_forge.config import DATA_DIR


def download_walmart_data():
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(
        competition="walmart-recruiting-store-sales-forecasting",
        path=DATA_DIR,
        force=True,
    )


def unzip_files(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        for file in zip_ref.namelist():
            if file.endswith(".zip"):
                nested_zip_path = os.path.join(extract_to, file)
                unzip_files(nested_zip_path, extract_to)
                os.remove(nested_zip_path)


def download_data():
    download_walmart_data()
    zip_path = DATA_DIR / "walmart-recruiting-store-sales-forecasting.zip"
    extract_to = DATA_DIR
    if os.path.exists(zip_path):
        unzip_files(zip_path, extract_to)
        # Delete the zip file after extraction
        os.remove(zip_path)


if __name__ == "__main__":
    download_data()
