MODEL_NAME="Alibaba-NLP/gte-Qwen2-7B-instruct"
HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download --local-dir $HOME/model $MODEL_NAME || exit 1

# Start vLLM service in background
uv run python -m vllm.entrypoints.openai.api_server \
--host 0.0.0.0 \
--model $HOME/model \
--max-model-len 3072 \
--task embed &

# Wait for vLLM to be ready by checking the health endpoint
echo "Waiting for vLLM service to be ready..."

while ! curl -s http://localhost:8000/health > /dev/null; do
  sleep 5
  echo "Still waiting for vLLM service..."
done

echo "vLLM service is ready!"
