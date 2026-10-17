# Cosmos DiffusionRenderer(Cosmos-Transfer1-DiffusionRenderer)를 서빙하는 이미지. DGX Spark(GB10, aarch64, CUDA 13)용이다.
# NGC PyTorch 이미지의 torch, Transformer Engine, CUDA를 그대로 쓴다. 모든 구성 요소는 공개 소스다.
FROM nvcr.io/nvidia/pytorch:25.09-py3

# ffmpeg는 영상 입출력에 쓴다. 나머지는 opencv(GUI 빌드)가 쓴다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libxcb1 libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# 의존성을 upstream 코드보다 먼저 설치한다. 패치를 바꿔도 오래 걸리는 이 레이어는 다시 빌드하지 않는다.
# NGC의 torch, torchvision, numpy, Transformer Engine이 바뀌지 않게 설치된 버전으로 고정한다.
COPY requirements.txt /tmp/requirements.txt
RUN pip list --format=freeze 2>/dev/null \
      | grep -E '^(torch|torchvision|numpy|transformer_engine)==' > /tmp/ngc-constraints.txt \
 && pip install --no-cache-dir -c /tmp/ngc-constraints.txt -r /tmp/requirements.txt

# nvdiffrast는 환경맵 샘플링에 쓴다. GB10의 compute capability로 빌드한다.
# nvdiffrast의 라이선스는 비상업 전용이다. 조건은 저장소 루트의 NOTICE에 있다.
ARG NVDIFFRAST_COMMIT=253ac4fcea7de5f396371124af597e6cc957bfae
ARG TORCH_CUDA_ARCH_LIST="12.1"
RUN TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" pip install --no-cache-dir --no-build-isolation \
      -c /tmp/ngc-constraints.txt \
      "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@${NVDIFFRAST_COMMIT}"

# 동작을 확인한 upstream 커밋으로 고정한다. 패치는 파일 이름 순서로 적용한다.
WORKDIR /opt/diffusion-renderer
ARG UPSTREAM_COMMIT=0f3e2dc435032ecbad654c2fc2153df85384b138
COPY patches/ /tmp/patches/
RUN git init -q . \
 && git remote add origin https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer.git \
 && git fetch -q --depth 1 origin "${UPSTREAM_COMMIT}" \
 && git checkout -q FETCH_HEAD \
 && git apply /tmp/patches/*.patch
ENV PYTHONPATH=/opt/diffusion-renderer

# HTTP 서버. upstream 코드와 분리해서 /opt/server에 둔다.
COPY server/requirements.txt /tmp/server-requirements.txt
RUN pip install --no-cache-dir -c /tmp/ngc-constraints.txt -r /tmp/server-requirements.txt
COPY server/app.py /opt/server/app.py

# upstream의 HDRI 예제 경로는 저장소 루트 기준 상대 경로다. 작업 디렉터리는 /opt/diffusion-renderer다.
EXPOSE 8000
CMD ["uvicorn", "app:app", "--app-dir", "/opt/server", "--host", "0.0.0.0", "--port", "8000"]
