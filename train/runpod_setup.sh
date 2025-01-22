# This the script I used to setup the environment for the training on runpod.io. You can use it to setup your own environment.
python --version

# Create and activate virtual environment
python -m venv /workspace/venv
source /workspace/venv/bin/activate

# Update pip and install requirements
pip install -U pip
pip install transformers datasets pandas numpy torch==2.1 trl tensorboard

# Create necessary directories
mkdir -p /workspace/data
mkdir -p /workspace/checkpoints

# Login to HuggingFace
echo "Please enter your HuggingFace token:"
read token
huggingface-cli login --token $token 