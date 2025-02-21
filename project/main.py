import subprocess
import time


imageName = "olama-test"
containerName = "olama_instance"

def build_image():
    print("Building the Docker image for Olama container...")
    command = f'docker build -t {imageName} .'
    subprocess.run(command, shell=True, check=True)

def start_docker_container():
    print("Starting terminal instance for Olama container...")
    command = f'start cmd /k docker run --gpus=all -it -p 9000:9000 --name {containerName} {imageName}'
    subprocess.Popen(command, shell=True)
    time.sleep(5)

def run_command():
    build_image()
    start_docker_container()

""" def fine_tune_model():
    print("Starting fine tuning of the model...")
    # Insert your model fine-tuning code here.
    # For example:
    #   - Load your dataset
    #   - Instantiate your model (or load a pretrained version)
    #   - Configure hyperparameters
    #   - Train the model
    #   - Save the updated model
    #
    # This is just a placeholder:
    for epoch in range(1, 6):
        print(f"Epoch {epoch}/5: training...")
        time.sleep(1)
    print("Model fine tuning completed successfully.") """
    
def stop_docker_container():
    print("Stopping Docker container...")
    subprocess.run(["docker", "stop", "olamma_instance"], check=True)
    subprocess.run(["docker", "rm", "olamma_instance"], check=True)
    print("Docker container stopped and removed.")

def main():
    try:
        run_command()
        """ fine_tune_model() """
    except subprocess.CalledProcessError as e:
        print("An error occurred while executing a Docker command:", e)
    finally:
        """ stop_docker_container() """

if __name__ == "__main__":
    main()