MODEL_NAME="ibm-granite/granite-3.3-8b-instruct"
MODEL_LOCATION="$HOME/deepseek_model"

HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download --local-dir "$MODEL_LOCATION" $MODEL_NAME || exit 1

# Start vLLM service in background
uv run python -m vllm.entrypoints.openai.api_server \
--host 0.0.0.0 \
--port 8001 \
--model "$MODEL_LOCATION" \
--gpu-memory-utilization 0.5 \
--max-model-len 32000 &

# Wait for vLLM to be ready by checking the health endpoint
echo "Waiting for vLLM service to be ready..."

while ! curl -s http://localhost:8001/health > /dev/null; do
  sleep 10
  echo "Still waiting for vLLM service..."
done

echo "vLLM service is ready!"
IP=$(curl -s https://ipinfo.io/ip)

echo "hit $IP:8001"
