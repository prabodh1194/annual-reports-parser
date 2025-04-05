## works on ec2-instance -- g6e.xlarge

sudo apt install python3.12

sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

curl -LsSf https://astral.sh/uv/install.sh | sh

bash

uv init --name qwen -p 3.12
uv add vllm transformers hf_transfer "huggingface-hub[cli]"
