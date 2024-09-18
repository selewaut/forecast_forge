import kaggle
import zipfile
import os


def download_walmart_data():
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(
        competition="walmart-recruiting-store-sales-forecasting",
        path="data/walmart_sales_forecasting",
        force=False,
    )


def unzip_files(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        for file in zip_ref.namelist():
            if file.endswith(".zip"):
                nested_zip_path = os.path.join(extract_to, file)
                nested_extract_to = os.path.join(extract_to, os.path.splitext(file)[0])
                os.makedirs(nested_extract_to, exist_ok=True)
                unzip_files(nested_zip_path, nested_extract_to)


if __name__ == "__main__":
    download_walmart_data()
    zip_path = (
        "data/walmart_sales_forecasting/walmart-recruiting-store-sales-forecasting.zip"
    )
    extract_to = "data/walmart_sales_forecasting"
    if os.path.exists(zip_path):
        unzip_files(zip_path, extract_to)
