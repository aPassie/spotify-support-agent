#!/usr/bin/env bash
# Download llama.cpp (CPU build) and the GGUF models, then start an OpenAI-compatible server.
#   scripts/setup_llm.sh gen    -> Qwen3-4B-Instruct-2507 Q4_K_M on http://127.0.0.1:8080/v1 (generator + silver labeller)
#   scripts/setup_llm.sh judge  -> gemma-3-4b-it Q4_K_M on http://127.0.0.1:8081/v1 (LLM judge)
# Any other OpenAI-compatible endpoint works too: set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
# (generator) and JUDGE_BASE_URL / JUDGE_MODEL (judge) instead of running this script.
set -euo pipefail
ROLE="${1:-gen}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLM="$ROOT/.llm"; mkdir -p "$LLM/models"
BUILD=b10874
if [ ! -x "$LLM/llama-$BUILD/llama-server" ]; then
  echo ">> downloading llama.cpp $BUILD (linux x64, CPU)"
  curl -sS -L -o "$LLM/llama-cpu.tar.gz" "https://github.com/ggml-org/llama.cpp/releases/download/$BUILD/llama-$BUILD-bin-ubuntu-x64.tar.gz"
  tar -xzf "$LLM/llama-cpu.tar.gz" -C "$LLM"
fi
case "$ROLE" in
  gen)   FILE=Qwen3-4B-Instruct-2507-Q4_K_M.gguf; URL="https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/$FILE"; PORT=8080 ;;
  judge) FILE=gemma-3-4b-it-Q4_K_M.gguf;          URL="https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/$FILE";          PORT=8081 ;;
  *) echo "usage: $0 gen|judge"; exit 1 ;;
esac
if [ ! -f "$LLM/models/$FILE" ]; then
  echo ">> downloading $FILE (~2.5 GB)"
  curl -sS -L -C - -o "$LLM/models/$FILE.part" "$URL" && mv "$LLM/models/$FILE.part" "$LLM/models/$FILE"
fi
THREADS="${LLM_THREADS:-$(( $(nproc) / 2 ))}"
echo ">> starting $ROLE server on :$PORT (threads=$THREADS, 4 parallel slots)"
exec "$LLM/llama-$BUILD/llama-server" -m "$LLM/models/$FILE" --host 127.0.0.1 --port "$PORT" -c 16384 -np 4 -t "$THREADS" --jinja --no-warmup
