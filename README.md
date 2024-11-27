
## Introduction

This repository contains code for the Walmart Sales Forecasting project. The project aims to forecast weekly sales for 45 Walmart stores located in different regions. The data includes historical sales data, holiday events, and store information.
Project contains framework to run and test multiple models in a spark environment. It also contains code to build spark docker image and run the spark container locally if needed.


## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/selewaut/forecast_forge.git
    cd forecast_forge
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
4. Install forecast_forge package
    ```sh
    pip install -e .
    ```

## Usage

### Data Preparation

The data is stored in the `data/` directory. The data is stored in the following files:

- `train.csv`: historical sales data for 45 Walmart stores
- `test.csv`: test data for forecasting
- `features.csv`: additional data related to the stores and regional activity
- `stores.csv`: store information

Data is originally downloaded using the following kaggle competition: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data.

For downloading data from kaggle, you need to have a kaggle account and kaggle API key. You can download the data using the following command:

1. Install kaggle package
    ```sh
    pip install kaggle
    ```
2. Generate API key from kaggle account and save it in `~/.kaggle/kaggle.json`



### Running the Code


1. Start container.

    ```sh
    make run
    ```
    This will start a spark container with the code mounted in the container. The container will be running in the background.

2. Run `univariate_weekly.py` script to train and test univariate weekly sales forecasting models.

    ```sh
    spark-submit --master local[*] src/forecast_forge/univariate_weekly.py
    ``` 

3. Results are saved in parquet format in evaluation_output path.