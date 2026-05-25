COMPOSE = docker compose
app ?= src/forecast_forge/univariate_weekly.py
APP ?= $(app)

.PHONY: build build-nc build-progress down run run-scaled run-d stop submit smoke logs ps

build:
	$(COMPOSE) build

build-nc:
	$(COMPOSE) build --no-cache

build-progress:
	$(COMPOSE) build --no-cache --progress=plain

down:
	$(COMPOSE) down --volumes --remove-orphans

run:
	$(COMPOSE) down --volumes --remove-orphans
	$(COMPOSE) up

run-scaled:
	$(COMPOSE) down --volumes --remove-orphans
	$(COMPOSE) up --scale spark-worker=3

run-d:
	$(COMPOSE) down --volumes --remove-orphans
	$(COMPOSE) up -d

stop:
	$(COMPOSE) stop

submit:
	docker exec da-spark-master spark-submit --master spark://spark-master:7077 --deploy-mode client /opt/spark/apps/$(APP)

smoke:
	docker exec da-spark-master spark-submit --master spark://spark-master:7077 --deploy-mode client /opt/spark/examples/src/main/python/pi.py 10

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps
