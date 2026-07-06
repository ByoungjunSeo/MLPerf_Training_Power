ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${FROM_IMAGE_NAME}

RUN git config --global user.name "a" && \
    git config --global user.email "a"

ENV PIP_CONSTRAINT=""

## 0. Pytorch Checkpoint size patch
WORKDIR /workspace/pytorch
COPY ./pytorch_ckpt.patch /workspace/pytorch/pytorch_ckpt.patch
RUN patch --directory=/usr/local/lib/python3.12/dist-packages -p1 < /workspace/pytorch/pytorch_ckpt.patch
WORKDIR /workspace/

## cuDNN (H100 OK)
RUN NVARCH=$(uname -m |grep x86_64 2>/dev/null || echo sbsa) && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH}/3bf863cc.pub > /etc/apt/trusted.gpg.d/cuda.asc && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && apt-get install -y libcudnn9-cuda-13=9.13.1.26-1 && \
    rm -rf /var/lib/apt/lists/*

ENV CUDNN_VERSION=9.13.1.26

RUN pip install numcodecs==0.13.1
RUN pip install nvidia-mathdx==25.1.1

## 1. Apex (H100 OK 그대로)
ARG APEX_REVISION=SKIP
ENV CUSTOM_APEX_REVISION ${APEX_REVISION}
ARG APEX_MAX_JOBS=4

RUN if [ "${APEX_REVISION}" != SKIP ]; then \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    git checkout ${APEX_REVISION} && \
    MAX_JOBS=${APEX_MAX_JOBS} NVCC_APPEND_FLAGS="--threads 8" \
    pip install -v --no-build-isolation --no-cache-dir \
      --config-settings "--build-option=--cpp_ext --cuda_ext --bnp --xentropy \
      --deprecated_fused_adam --deprecated_fused_lamb --fast_multihead_attn \
      --distributed_lamb --fast_layer_norm --distributed_adam --fmha \
      --fast_bottleneck --nccl_p2p --peer_memory --permutation_search \
      --focal_loss --fused_conv_bias_relu --index_mul_2d --cudnn_gbn \
      --group_norm" . ; \
    fi

## 2. Transformer Engine (🔥 핵심 수정 🔥)
ARG TE_REVISION=release_v2.8
ENV CUSTOM_TE_REVISION ${TE_REVISION}

RUN if [ "${TE_REVISION}" != SKIP ]; then \
    pip uninstall -y transformer-engine && \
    git clone https://github.com/NVIDIA/TransformerEngine.git transformerengine && \
    cd transformerengine && \
    git checkout ${TE_REVISION} && \
    echo $(git rev-parse HEAD) > /TE_COMMIT_HASH.env && \
    git submodule update --init --recursive && \
    git -C 3rdparty/cudnn-frontend checkout 80a8e4af4d89d33a2c59d51fcf9fda1c9d368cd4 && \
    git -C 3rdparty/googletest checkout f8d7d77c06936315286eb55f8de22cd23c188571 && \
    git -C 3rdparty/cutlass checkout 57e3cfb47a2d9e0d46eb6335c3dc411498efa198 && \
    \
    # ✅ H100 PCIe = SM90
    NVTE_CUDA_ARCHS="90" \
    NVTE_UB_WITH_MPI=1 \
    NVTE_FRAMEWORK=pytorch \
    MPI_HOME=/usr/local/mpi \
    pip install --no-build-isolation . ; \
    fi

## 3. NeMo
ARG NEMO_REVISION=25.09-alpha.rc1
ENV CUSTOM_NEMO_REVISION ${NEMO_REVISION}

RUN git clone https://github.com/NVIDIA/NeMo.git && \
    cd NeMo && \
    git checkout ${NEMO_REVISION} && \
    echo $(git rev-parse HEAD) > /NEMO_COMMIT_HASH.env && \
    pip uninstall -y nemo-toolkit sacrebleu && \
    pip install "cython<3.0.0" && \
    pip install -e ".[llm]"

## 3.1 NeMo-Run
ARG NEMORUN_REVISION=v0.5.0
RUN git clone https://github.com/NVIDIA/NeMo-Run.git && \
    cd NeMo-Run && \
    git checkout ${NEMORUN_REVISION} && \
    pip install -e .

## 4. Megatron-core
ARG MCORE_REVISION=25.09-alpha.rc1
RUN pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM && \
    cd Megatron-LM && \
    git checkout ${MCORE_REVISION} && \
    echo $(git rev-parse HEAD) > /MCORE_COMMIT_HASH.env && \
    pip install -e . && \
    cd megatron/core/datasets && make

ENV PYTHONPATH="${PYTHONPATH}:/workspace/Megatron-LM"

## 5. Benchmark deps
RUN pip uninstall transformers -y
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

## Benchmark code
WORKDIR /workspace/llm
COPY . .
RUN cd /workspace/llm/embedding_lib && pip install --force-reinstall --no-deps .
ENV PYTHONPATH="/workspace/llm:/workspace/NeMo:${PYTHONPATH}"

RUN pip uninstall -y bitsandbytes

