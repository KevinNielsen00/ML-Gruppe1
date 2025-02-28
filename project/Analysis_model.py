#%% imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from datasets import load_dataset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import torch
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

#%% Load dataset
dataset_path = "G:/Projects/ML-Gruppe1/project/datacenter/Code-Golang-QA-2k/dataset.json"

with open(dataset_path, 'r') as file:
    dataset = json.load(file)
print(dataset[:5])
#%% Test if model is loaded correctly
# model path
model_dir = "models/DeepSeek-R1-Distill-Qwen-1.5B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.float16
).to(device)

print("Model Loaded Successfully!")

#%% Tokenization Analysis
#Tokenization is the process of converting text into a sequence of tokens
def analyze_tokens(prompt):
    tokens = tokenizer.tokenize(prompt)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")

# Add any promt to see the tokenization process
analyze_tokens("Is the earth flat?")

#%% Speed Benchmarking for GPU
def measure_speed(prompt, device):
    if device == "cuda" and not torch.cuda.is_available():
        print("No GPU found, running only on CPU.")
        device = "cpu"

    model.to(device) 
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()
    _ = model.generate(input_ids, max_length=200, top_p=0.95)

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"{device.upper()} Generation Time: {elapsed_time:.2f} seconds")
    return elapsed_time

# Run benchmark CPU
prompt_text = "What is the meaning of life?"
devices = ["cpu"]
times = [measure_speed(prompt_text, "cpu")]

if torch.cuda.is_available():
    devices.append("gpu")
    times.append(measure_speed(prompt_text, "cuda"))
    speedup = times[0] / times[1]
    print(f"Speedup Factor: GPU is {speedup:.2f}x faster than CPU!")

# Plot results
plt.figure(figsize=(6, 4))
plt.bar(devices, times, color=["red", "blue"] if len(devices) > 1 else ["red"])
plt.ylabel("Time (seconds)")
plt.title("GPU vs. CPU Text Generation Speed")
plt.show()

#%% sentiment analysis
#This one is mostly used to determine the sentiment of a social media text, whether it is positive, negative, or neutral.
# our answers in the dataset are more formal and technical therefore its neutral or more inclined towards positive language. the answers are informative, clear, and not negative in sentiment.
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(answer):
    score = analyzer.polarity_scores(answer)
    if score['compound'] > 0.05:
        return 'Positive'
    elif score['compound'] < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Test with answers from the dataset
for entry in dataset[:5]:
    sentiment = analyze_sentiment(entry['answer'])
    print(f"Q: {entry['question']}\nSentiment: {sentiment}\n")

#%% Clustering
# Each answer is assigned a cluster label (0, 1, 2), and these labels represent the cluster the answer belongs to.
answers = [entry['answer'] for entry in dataset]

# Converts answers to numerical form using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(answers)

# Fit a KMeans model to cluster answers
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

# Assign clusters to answers
for i, label in enumerate(kmeans.labels_):
    print(f"Answer: {answers[i]} \nCluster: {label}\n")

#%% visualizing of the clusters
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import json

dataset_path = "G:/Projects/ML-Gruppe1/project/datacenter/Code-Golang-QA-2k/dataset.json"

with open(dataset_path, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

# Extract answers
answers = [entry["answer"] for entry in dataset if "answer" in entry]

# Convert answers to numerical representation using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(answers)

# Apply KMeans clustering
n_clusters = 3  # Change this to experiment with different numbers of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_

# Reduce dimensions using t-SNE
X_embedded = TSNE(n_components=2, perplexity=3, random_state=42).fit_transform(X.toarray())

# Plot the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=labels, palette='viridis', s=100)
plt.title("Text Clusters Visualized with t-SNE")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Cluster")
plt.show()

# how it should look for our model if we where able to run the training with tensorboard
def load_metrics_from_tensorboard(log_dir):
    #Reads metrics from TensorBoard log file
    event_files = [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]
    if not os.path.exists(log_dir):
        print("Log directory does not exist.")
        return [], [], [], [], []
    log_file = os.path.join(log_dir, event_files[0]) 
    ea = event_accumulator.EventAccumulator(log_file)
    ea.Reload()

    # Extract metrics we want to plot
    losses = [e.value for e in ea.Scalars('loss')] if 'loss' in ea.Tags()['scalars'] else []
    eval_losses = [e.value for e in ea.Scalars('eval_loss')] if 'eval_loss' in ea.Tags()['scalars'] else []
    accuracies = [e.value for e in ea.Scalars('accuracy')] if 'accuracy' in ea.Tags()['scalars'] else []
    precisions = [e.value for e in ea.Scalars('precision')] if 'precision' in ea.Tags()['scalars'] else []
    recalls = [e.value for e in ea.Scalars('recall')] if 'recall' in ea.Tags()['scalars'] else []

    return losses, eval_losses, accuracies, precisions, recalls

def plot_metrics_curve(losses, eval_losses, accuracies, precisions, recalls):
    """Plot trænings- og evalueringsmetrics."""
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.plot(losses, label="Training Loss", color="blue")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid()

    plt.subplot(2, 3, 2)
    plt.plot(eval_losses, label="Validation Loss", color="red")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()
    plt.grid()

    plt.subplot(2, 3, 3)
    plt.plot(accuracies, label="Accuracy", color="green")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.grid()

    plt.subplot(2, 3, 4)
    plt.plot(precisions, label="Precision", color="orange")
    plt.xlabel("Steps")
    plt.ylabel("Precision")
    plt.title("Training Precision")
    plt.legend()
    plt.grid()

    plt.subplot(2, 3, 5)
    plt.plot(recalls, label="Recall", color="purple")
    plt.xlabel("Steps")
    plt.ylabel("Recall")
    plt.title("Training Recall")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


# %% Simulated the one due to not having enough allocated memory to beable to run it with the data needed
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Simulated epoch values
epochs = np.arange(1, 51)

# Generate simulated values for loss, accuracy, precision, and recall
loss_values = np.exp(-0.1 * epochs) + np.random.normal(0, 0.02, size=epochs.shape)
accuracy_values = 0.5 + 0.5 * (1 - np.exp(-0.1 * epochs)) + np.random.normal(0, 0.01, size=epochs.shape)
precision_values = 0.4 + 0.6 * (1 - np.exp(-0.1 * epochs)) + np.random.normal(0, 0.01, size=epochs.shape)
recall_values = 0.3 + 0.7 * (1 - np.exp(-0.1 * epochs)) + np.random.normal(0, 0.01, size=epochs.shape)

# Define exponential fitting function
def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Fit exponential curves
loss_params, _ = curve_fit(exp_func, epochs, loss_values)
accuracy_params, _ = curve_fit(exp_func, epochs, accuracy_values)
precision_params, _ = curve_fit(exp_func, epochs, precision_values)
recall_params, _ = curve_fit(exp_func, epochs, recall_values)

# Predict values using fitted parameters
y_loss_pred = exp_func(epochs, *loss_params)
y_accuracy_pred = exp_func(epochs, *accuracy_params)
y_precision_pred = exp_func(epochs, *precision_params)
y_recall_pred = exp_func(epochs, *recall_params)

# Plot results
plt.figure(figsize=(10, 6))

plt.plot(epochs, loss_values, 'bo', label="Actual Loss")
plt.plot(epochs, y_loss_pred, 'b--', label="Loss Fit")

plt.plot(epochs, accuracy_values, 'go', label="Actual Accuracy")
plt.plot(epochs, y_accuracy_pred, 'g--', label="Accuracy Fit")

plt.plot(epochs, precision_values, 'ro', label="Actual Precision")
plt.plot(epochs, y_precision_pred, 'r--', label="Precision Fit")

plt.plot(epochs, recall_values, 'mo', label="Actual Recall")
plt.plot(epochs, y_recall_pred, 'm--', label="Recall Fit")

plt.xlabel("Epochs")
plt.ylabel("Metrics")
plt.title("Simulated Model Performance Over Epochs")
plt.legend()
plt.show()


# %% linear regression simulated loss
# Simulated loss data
epochs = np.arange(1, 51)
loss_values = np.exp(-0.1 * epochs) + np.random.normal(0, 0.02, size=epochs.shape)

X_train = epochs.reshape(-1, 1)
y_train = loss_values

# Train a regression model to fit the loss curve
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_train)

# Plot actual loss vs. regression fit
plt.figure(figsize=(8, 5))
plt.scatter(epochs, loss_values, label="Actual Loss", color="blue")
plt.plot(epochs, y_pred, label="Regression Fit", color="red", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Regression Analysis")
plt.legend()
plt.show()

# Print regression equation
print(f"Regression equation: Loss = {reg_model.coef_[0]:.4f} * Epoch + {reg_model.intercept_:.4f}")
