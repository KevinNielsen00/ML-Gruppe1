FROM ubuntu:latest

RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://ollama.com/install.sh | sh

# Copy your local model into the image.
COPY models/deepseek-coder-7b-instruct-v1.5 /model

# Create an entrypoint script that starts the ollama service and then loads your local model.
RUN echo '#!/bin/sh\n\
ollama start &\n\
sleep 5\n\
ollama run /model' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

EXPOSE 9000

CMD ["/entrypoint.sh"]