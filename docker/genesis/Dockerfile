# S-ARM101 + Genesis World for Linux NVIDIA batch sim / recording.
# Build:  docker build -t sarm-hand-genesis -f docker/genesis/Dockerfile .
# Run:    docker run --gpus all -v $(pwd)/data:/workspace/data sarm-hand-genesis record-sim --headless

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install uv for reproducible sarm-hand install
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY . /workspace/hand
WORKDIR /workspace/hand

RUN uv sync --extra genesis && \
    ./scripts/fetch_so101_assets.sh

ENV VIRTUAL_ENV=/workspace/hand/.venv
ENV PATH="/workspace/hand/.venv/bin:${PATH}"

ENTRYPOINT ["sarm-hand"]
CMD ["genesis-spike", "--headless"]
