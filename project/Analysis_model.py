#%% imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from datasets import load_dataset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import json
import torch
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

#%% Load dataset
dataset_path = "G:/Projects/ML-Gruppe1/project/datacenter/Code-Golang-QA-2k/dataset.json"

with open(dataset_path, 'r') as file:
    dataset = json.load(file)
print(dataset[:5])
#%% Test if model is loaded correctly
# model path
model_dir = "project/DeepSeek-R1-Distill-Qwen-1.5B"

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

# %% Exponential Regression simulated loss
def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

params, _ = curve_fit(exp_func, epochs, loss_values)
params[2] = max(params[2], 0)

y_exp_predictions = exp_func(epochs, *params)

# Plot results
plt.scatter(epochs, loss_values, label="Actual Loss", color="blue")
plt.plot(epochs, y_exp_predictions, label="Exponential Fit", color="red", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Exponential Regression for Loss Trend")
plt.legend()
plt.show()

print(f"Exponential equation: Loss = {params[0]:.4f} * exp(-{params[1]:.4f} * Epoch) + {params[2]:.4f}")

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
# Each answer is assigned a cluster label (0, 1, 2, ...), and these labels represent the cluster the answer belongs to.
answers = [entry['answer'] for entry in dataset]

# Convert answers to numerical form using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(answers)

# Fit a KMeans model to cluster answers
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Assign clusters to answers
for i, label in enumerate(kmeans.labels_):
    print(f"Answer: {answers[i]} \nCluster: {label}\n")

#%% 
