# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps — cached by apt across builds
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Python deps — cached by pip across builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install \
        jupyterlab \
        ipywidgets \
        ir-datasets \
        ranx \
        opensearch-py \
        tqdm \
        sentence-transformers \
        openai \
        ipykernel \
        debugpy \
        pytrec-eval

# Register a python3 kernelspec inside the image
RUN python -m ipykernel install --name python3 --display-name "Python 3" --sys-prefix || true

WORKDIR /workspace

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

