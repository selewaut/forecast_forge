FROM python:3.13-slim-bookworm AS spark-base

ARG SPARK_VERSION=4.1.1
ARG UV_VERSION=0.11.9

ENV SPARK_HOME=/opt/spark
ENV HADOOP_HOME=/opt/hadoop
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="/workspace/.venv/bin:/opt/spark/sbin:/opt/spark/bin:${JAVA_HOME}/bin:${PATH}"
ENV PYSPARK_PYTHON=/workspace/.venv/bin/python
ENV PYSPARK_DRIVER_PYTHON=/workspace/.venv/bin/python
ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      build-essential \
      ca-certificates \
      curl \
      openjdk-17-jdk \
      procps \
      rsync \
      ssh \
      tini \
      unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p "${HADOOP_HOME}" "${SPARK_HOME}" /workspace /opt/java && \
    ln -s /usr/lib/jvm/java-17-openjdk-* "${JAVA_HOME}"

WORKDIR "${SPARK_HOME}"

RUN curl -fL "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz" \
      -o "spark-${SPARK_VERSION}-bin-hadoop3.tgz" && \
    tar xzf "spark-${SPARK_VERSION}-bin-hadoop3.tgz" --directory "${SPARK_HOME}" --strip-components 1 && \
    rm "spark-${SPARK_VERSION}-bin-hadoop3.tgz"

FROM spark-base AS pyspark

ARG UV_VERSION=0.11.9

WORKDIR /workspace

RUN pip install --no-cache-dir "uv==${UV_VERSION}"

COPY pyproject.toml uv.lock README.md LICENSE .python-version ./
COPY src ./src

RUN uv sync --frozen --no-dev

COPY spark-setup/conf/spark-defaults.conf "${SPARK_HOME}/conf/spark-defaults.conf"
COPY --chmod=755 spark-setup/entrypoint.sh "${SPARK_HOME}/entrypoint.sh"

RUN mkdir -p "${SPARK_HOME}/spark-events" && \
    chmod +x "${SPARK_HOME}/entrypoint.sh" && \
    test -x "${SPARK_HOME}/entrypoint.sh"

WORKDIR "${SPARK_HOME}"

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/spark/entrypoint.sh"]
