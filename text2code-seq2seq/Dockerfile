# Dockerfile for Text-to-Code Generation Project
# Provides complete environment for training and evaluation

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints results/loss_curves results/attention_viz report

# Set environment variables for reproducibility
ENV PYTHONHASHSEED=42
ENV CUDA_LAUNCH_BLOCKING=1

# Expose port for Jupyter (optional)
EXPOSE 8888

# Default command
CMD ["/bin/bash"]

# Usage instructions:
# 
# Build:
#   docker build -t text2code-seq2seq .
#
# Run (CPU):
#   docker run -it --rm -v $(pwd)/checkpoints:/workspace/checkpoints text2code-seq2seq
#
# Run (GPU):
#   docker run -it --rm --gpus all -v $(pwd)/checkpoints:/workspace/checkpoints text2code-seq2seq
#
# Run training:
#   docker run -it --rm --gpus all -v $(pwd)/checkpoints:/workspace/checkpoints text2code-seq2seq python train.py
#
# Run evaluation:
#   docker run -it --rm --gpus all -v $(pwd)/checkpoints:/workspace/checkpoints -v $(pwd)/results:/workspace/results text2code-seq2seq python evaluate.py
#
# Interactive mode with Jupyter:
#   docker run -it --rm -p 8888:8888 --gpus all text2code-seq2seq jupyter notebook --ip=0.0.0.0 --allow-root
