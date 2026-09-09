#!/usr/bin/env bash
# Download llama.cpp (CPU build) and the GGUF models, then start an OpenAI-compatible server.
#   scripts/setup_llm.sh gen    -> Qwen3-4B-Instruct-2507 Q4_K_M on http://127.0.0.1:8080/v1 (generator + silver labeller)
#   scripts/setup_llm.sh judge  -> gemma-3-4b-it Q4_K_M on http://127.0.0.1:8081/v1 (LLM judge)
#   scripts/setup_llm.sh embed  -> bge-small-en-v1.5 on http://127.0.0.1:8082/v1 (dense retrieval ablation)
# Any other OpenAI-compatible endpoint works too: set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
# (generator) and JUDGE_BASE_URL / JUDGE_MODEL (judge) instead of running this script.
set -euo pipefail
ROLE="${1:-gen}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLM="$ROOT/.llm"; mkdir -p "$LLM/models"
BUILD=b10874
# GPU_BACKEND=vulkan (default) gets the prebuilt Vulkan binaries, which work on NVIDIA and AMD
# with just the normal driver plus a Vulkan loader. llama.cpp ships no prebuilt CUDA build for
# Linux (Windows only), so CUDA means building from source; Vulkan was ~7x CPU here and enough.
# Set GPU_BACKEND=cpu to force the CPU build.
BACKEND="${GPU_BACKEND:-vulkan}"
if [ "$BACKEND" = "vulkan" ] && [ ! -x "$LLM/vk/llama-$BUILD/llama-server" ]; then
  echo ">> downloading llama.cpp $BUILD (linux x64, Vulkan)"
  curl -sS -L -o "$LLM/llama-vulkan.tar.gz" "https://github.com/ggml-org/llama.cpp/releases/download/$BUILD/llama-$BUILD-bin-ubuntu-vulkan-x64.tar.gz"
  mkdir -p "$LLM/vk" && tar -xzf "$LLM/llama-vulkan.tar.gz" -C "$LLM/vk"
fi
if [ ! -x "$LLM/llama-$BUILD/llama-server" ] && [ ! -x "$LLM/vk/llama-$BUILD/llama-server" ]; then
  echo ">> downloading llama.cpp $BUILD (linux x64, CPU)"
  curl -sS -L -o "$LLM/llama-cpu.tar.gz" "https://github.com/ggml-org/llama.cpp/releases/download/$BUILD/llama-$BUILD-bin-ubuntu-x64.tar.gz"
  tar -xzf "$LLM/llama-cpu.tar.gz" -C "$LLM"
fi
case "$ROLE" in
  gen)   FILE=Qwen3-4B-Instruct-2507-Q4_K_M.gguf; URL="https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/$FILE"; PORT=8080 ;;
  judge) FILE=gemma-3-4b-it-Q4_K_M.gguf;          URL="https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/$FILE";          PORT=8081 ;;
  embed) FILE=bge-small-en-v1.5-f16.gguf;         URL="https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/$FILE"; PORT=8082 ;;
  *) echo "usage: $0 gen|judge|embed"; exit 1 ;;
esac
if [ ! -f "$LLM/models/$FILE" ]; then
  echo ">> downloading $FILE (~2.5 GB)"
  curl -sS -L -C - -o "$LLM/models/$FILE.part" "$URL" && mv "$LLM/models/$FILE.part" "$LLM/models/$FILE"
fi
THREADS="${LLM_THREADS:-$(( $(nproc) / 2 ))}"
# Prefer a GPU build when one is present: the Vulkan binaries work on both NVIDIA and AMD with
# only the normal driver installed. GGML_VK_VISIBLE_DEVICES pins to one device on hybrid laptops.
BINDIR="$LLM/llama-$BUILD"
if [ -x "$LLM/vk/llama-$BUILD/llama-server" ]; then BINDIR="$LLM/vk/llama-$BUILD"; fi
GPU_ARGS=""
if [ "$BINDIR" != "$LLM/llama-$BUILD" ]; then GPU_ARGS="-ngl 99"; fi
export LD_LIBRARY_PATH="$BINDIR:${LD_LIBRARY_PATH:-}"
echo ">> starting $ROLE server on :$PORT (bin=$(basename "$(dirname "$BINDIR")")/$(basename "$BINDIR"), threads=$THREADS)"
if [ "$ROLE" = "embed" ]; then
  # --no-mmap matters: with layers offloaded it cut host RAM from 7.4 GB to 0.5 GB here.
  exec "$BINDIR/llama-server" -m "$LLM/models/$FILE" --host 127.0.0.1 --port "$PORT"     --embeddings --pooling cls -c 512 -b 512 -ub 512 -np 4 -t "$THREADS" --no-mmap $GPU_ARGS
fi
exec "$BINDIR/llama-server" -m "$LLM/models/$FILE" --host 127.0.0.1 --port "$PORT"   -c 8192 -np 4 -t "$THREADS" -fa on --no-mmap --jinja --no-warmup $GPU_ARGS
