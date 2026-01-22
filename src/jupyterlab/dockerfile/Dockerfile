FROM python:3.11-slim

# System deps (safe for ir_datasets, ranx)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    jupyterlab 
#    ir-datasets \
#    ranx \
#    opensearch-py \
#    tqdm

WORKDIR /workspace

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]


