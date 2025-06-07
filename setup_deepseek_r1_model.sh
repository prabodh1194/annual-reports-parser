MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MODEL_LOCATION="$HOME/deepseek_model"

HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download --local-dir "$MODEL_LOCATION" $MODEL_NAME || exit 1

# Start vLLM service in background
uv run python -m vllm.entrypoints.openai.api_server \
--host 0.0.0.0 \
--model "$MODEL_LOCATION" \
--max-model-len 4096 &

# Wait for vLLM to be ready by checking the health endpoint
echo "Waiting for vLLM service to be ready..."

while ! curl -s http://localhost:8001/health > /dev/null; do
  sleep 10
  echo "Still waiting for vLLM service..."
done

echo "vLLM service is ready!"
IP=$(curl -s https://ipinfo.io/ip)

echo "hit $IP:8001"
