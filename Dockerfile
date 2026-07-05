# LeAgents runtime image (issue #1) — everything that took a debugging
# session to install natively, baked in: EGL headers for LIBERO's egl-probe
# build, CMake < 4 (Ubuntu 24.04 ships 3.28, so no CMAKE_POLICY_VERSION_
# MINIMUM override), libero's one-time non-interactive init, and headless
# sim env vars. GPU via nvidia-container-toolkit:
#
#   docker build -t leagents .
#   docker run --rm --gpus all -v ./runs:/app/runs leagents \
#     leagents run -c configs/m0_libero.yaml
#
# Secrets are NEVER baked in — pass at runtime (--env-file or compose).
# INSTALL_LEROBOT=0 builds a slim loop-only image (CI uses it: the full
# torch/CUDA stack doesn't fit hosted-runner disks).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG INSTALL_LEROBOT=1

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    SDL_VIDEODRIVER=dummy \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv python3-pip git ffmpeg \
        cmake build-essential \
        libegl1-mesa-dev libgles2-mesa-dev libglvnd0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY . .

RUN pip install -e ".[dev,dash,llm]" \
    && if [ "$INSTALL_LEROBOT" = "1" ]; then \
         pip install -e ".[lerobot]" \
         && echo N | python -c "import libero.libero"; \
       fi

EXPOSE 8321
CMD ["leagents", "--help"]
