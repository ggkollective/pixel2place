# Cosmos3-Nano Reasoner를 서빙하는 vLLM 이미지. DGX Spark(GB10, aarch64, CUDA 13)용이다.
# 모든 구성 요소는 공개 소스다.
FROM nvcr.io/nvidia/pytorch:25.09-py3

# 이미지·비디오 처리 의존성이 쓰는 그래픽 라이브러리.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# cosmos-framework의 pyproject는 uv 0.11.3 이상을 요구한다.
RUN pip install --no-cache-dir "uv>=0.11.3"

WORKDIR /app

# 동작을 확인한 커밋으로 고정한다. 두 노드가 같은 이미지를 쓰므로 한 번만 빌드한다.
ARG VLLM_COMMIT=598d51153a15ff4a803037dade1c4ea43d1ea9bc
ARG COSMOS_FRAMEWORK_COMMIT=dff6966446f2325f69e9a65c396908019b721df1
# GB10의 compute capability.
ARG TORCH_CUDA_ARCH_LIST="12.1"
RUN uv venv --python 3.12 --seed --managed-python .venv \
 && . .venv/bin/activate \
 && TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" uv pip install --torch-backend=cu130 \
    "vllm @ git+https://github.com/vllm-project/vllm.git@${VLLM_COMMIT}" \
 && TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" uv pip install --torch-backend=cu130 \
    "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git@${COSMOS_FRAMEWORK_COMMIT}#subdirectory=packages/vllm-cosmos3"

ENV PATH=/app/.venv/bin:$PATH
# DeepGEMM은 이 CUDA 조합용 빌드가 없다.
ENV VLLM_USE_DEEP_GEMM=0

EXPOSE 8000
