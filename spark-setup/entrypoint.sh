#!/bin/bash
set -euo pipefail

SPARK_WORKLOAD=${1:-}
echo "SPARK_WORKLOAD: $SPARK_WORKLOAD"

mkdir -p "${SPARK_HOME}/spark-events"

if [ "$SPARK_WORKLOAD" == "master" ];
then
  exec start-master.sh -p 7077
elif [ "$SPARK_WORKLOAD" == "worker" ];
then
  exec start-worker.sh spark://spark-master:7077
elif [ "$SPARK_WORKLOAD" == "history" ]
then
  exec start-history-server.sh
else
  exec "$@"
fi
