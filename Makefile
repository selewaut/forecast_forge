DATA_DIR = data/walmart_sales_forecasting
DATA_FILES = $(DATA_DIR)/train.csv $(DATA_DIR)/test.csv $(DATA_DIR)/features.csv $(DATA_DIR)/stores.csv
OUTPUT_DIR = weekly_evaluation_output
MODEL ?=

.PHONY: setup check data run clean help

setup: ## Install project dependencies into the local uv environment.
	uv sync
	@echo "✓ Environment ready"

check: ## Validate Java, Kaggle credentials, and local dataset presence.
	@echo "--- Checking prerequisites ---"
	@version=$$(direnv exec . java -version 2>&1) && \
		echo "✓ Java: $$(echo "$$version" | head -1)" || { \
		echo "✗ Java not functional — run: direnv allow"; \
		exit 1; \
	}
	@if [ -f ~/.kaggle/kaggle.json ] || [ -f ~/.kaggle/access_token ] || [ -n "$$KAGGLE_USERNAME" ]; then \
		echo "✓ Kaggle credentials: found"; \
	else \
		echo "✗ Kaggle credentials missing — place at ~/.kaggle/kaggle.json or ~/.kaggle/access_token"; \
		exit 1; \
	fi
	@if [ -f $(DATA_DIR)/train.csv ]; then \
		echo "✓ Data: found"; \
	else \
		echo "⚠ Data not found — run: make data"; \
	fi
	@echo "--- All checks passed ---"

data: $(DATA_FILES) ## Download and verify the Walmart forecasting dataset.
	@echo "✓ Verified dataset files in $(DATA_DIR)"

$(DATA_FILES):
	@echo "Downloading Walmart dataset from Kaggle..."
	uv run python -c "from forecast_forge.data import download_data; download_data()"
	@echo "✓ Data downloaded to $(DATA_DIR)"

run: ## Run the local Spark forecasting pipeline. Optionally pass MODEL=<model>.
	direnv exec . uv run spark-submit --master 'local[*]' src/forecast_forge/univariate_weekly.py $(if $(MODEL),--model $(MODEL),)

clean: ## Remove generated forecast, Spark, and MLflow artifact outputs.
	rm -rf $(OUTPUT_DIR) mlruns
	@echo "✓ Removed generated output directories"

help: ## Show available make targets.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'
