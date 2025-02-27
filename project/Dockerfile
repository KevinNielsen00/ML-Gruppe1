FROM python:3.8-slim

WORKDIR /app

# Install system dependencies needed for scipy and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the required files
COPY start_up.py ./start_script.py
COPY requirements.txt ./requirements.txt    

# Create directories for models
RUN mkdir -p ./models
RUN mkdir -p ./finetuned_lora_model

# Copy model files
COPY models/DeepSeek-R1-Distill-Qwen-1.5B ./models/DeepSeek-R1-Distill-Qwen-1.5B
COPY finetuned_lora_model ./finetuned_lora_model

# Install dependencies with verbosity to see any issues
RUN pip install --upgrade pip && \
    pip install --verbose -r requirements.txt

# Set environment variables for GPU support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 9000

CMD ["python", "start_script.py"]