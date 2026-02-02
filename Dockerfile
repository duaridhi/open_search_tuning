FROM python:3.11-slim

# System deps (safe for ir_datasets, ranx)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --upgrade pip setuptools wheel

# Install jupyterlab and project dependencies plus ipykernel and debugpy
RUN pip install \
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

# Register a python3 kernelspec inside the image (sys-prefix keeps it inside the environment)
RUN python -m ipykernel install --name python3 --display-name "Python 3" --sys-prefix || true

WORKDIR /workspace

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

