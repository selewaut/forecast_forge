import kaggle


def download_walmart_data():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset="walmart-recruiting-store-sales-forecasting",
        path="data/walmart_sales_forecasting",
        unzip=True,
    )


if __name__ == "__main__":
    download_walmart_data()
