
## Introduction

This repository contains code for the Walmart Sales Forecasting project. The project aims to forecast weekly sales for 45 Walmart stores located in different regions. The data includes historical sales data, holiday events, and store information.
Project contains framework to run and test multiple models in a spark environment. It also contains code to build spark docker image and run the spark container locally if needed.


## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/selewaut/forecast_forge.git
    cd forecast_forge
    ```

2. Install `uv` and Python 3.13:
    ```sh
    brew install uv
    uv python install 3.13
    ```

3. Install Java 17 for local Spark execution:
    ```sh
    brew install openjdk@17
    ```

4. Install the project dependencies:
    ```sh
    uv sync
    ```

On Linux, install OpenJDK 17 with your system package manager. On Windows, prefer WSL for local development and install OpenJDK 17 inside the WSL distribution.

### Project Environment

This repository includes a `.envrc` that sets Homebrew OpenJDK 17 for this project:

```sh
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export PATH="$JAVA_HOME/bin:$PATH"
```

Install and enable `direnv` on macOS:

```sh
brew install direnv
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
source ~/.zshrc
direnv allow
```

The shell hook is global, but the environment values are project-specific. The hook only teaches zsh to ask `direnv` whether the current directory has an approved `.envrc`. The Java 17 values above load when the shell enters this repository and unload when it leaves.

After setup, verify the project environment with:

```sh
direnv exec . sh -c 'echo JAVA_HOME=$JAVA_HOME && java -version'
```

Without `direnv`, export the same `JAVA_HOME` and `PATH` values manually before running local Spark commands:

```sh
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export PATH="$JAVA_HOME/bin:$PATH"
```

## Local Spark Smoke Test

Validate local PySpark execution with:

```sh
JAVA_HOME=/opt/homebrew/opt/openjdk@17 PATH=/opt/homebrew/opt/openjdk@17/bin:$PATH uv run python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.master('local[*]').appName('forecast-forge-smoke').getOrCreate(); spark.range(1).show(); spark.stop()"
```

If Java 17 is already configured in your shell, or `direnv allow` has loaded the project `.envrc`, this shorter command should work:

```sh
uv run python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.master('local[*]').getOrCreate(); spark.range(1).show(); spark.stop()"
```

For non-interactive shells or CI commands, prefer either `direnv exec . <command>` or explicit `JAVA_HOME`/`PATH` values.

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

#### Local PySpark

Run local Spark jobs through `uv` when a cluster is not needed:

```sh
uv run spark-submit --master 'local[*]' src/forecast_forge/univariate_weekly.py
```

#### Docker Spark Cluster

1. Move to the Spark setup directory.

    ```sh
    cd spark-setup
    ```
2. Build the image.

    ```sh
    make build
    ```

3. Start the cluster in the background.

    ```sh
    make run-d
    ```

4. Run the cluster smoke test.

    ```sh
    make smoke
    ```

5. Submit `univariate_weekly.py` to the Spark master.

    ```sh
    make submit app=src/forecast_forge/univariate_weekly.py
    ``` 

Results are saved in parquet format in evaluation_output path.

The Spark master UI is available at http://localhost:9090. The history server is available at http://localhost:18080.
