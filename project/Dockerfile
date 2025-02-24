FROM ubuntu:latest

RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app 

COPY models/Llama-3.2-3B-Q5_K_M.gguf ./Llama-3.2-3B-Q5_K_M.gguf

# Updated model configuration file with the correct model path
RUN echo "FROM /app/Llama-3.2-3B-Q5_K_M.gguf" > Modelfile

# Create an entrypoint script that starts the ollama service and then loads your local model.
# ol
RUN echo '#!/bin/sh\n\
ollama start &\n\
sleep 5\n\
ollama create llama3.2-test -f ./Modelfile\n\
ollama run llama3.2-test' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

EXPOSE 9000

CMD ["/entrypoint.sh"]