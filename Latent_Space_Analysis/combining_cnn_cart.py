import os
import re
import pandas as pd
from collections import defaultdict

# Define your folders
base_dir = os.path.expanduser("~/Downloads")
dataset_types = ["Semi_imbalanced", "Highly_Imbalanced", "Balanced"]
sample_sizes = ["1000", "2000", "3000", "4000", "5000"]
model_folders = [
    "Diffusion", "DVAE_gaussian", "DVAE_contrast", "VQVAE", "DVAE_blur", 
    "DVAE_brightness", "GAN", "DVAE_salt_pepper", "VAE", "BIGGAN", 
    "DVAE_rotation", "CLIP-guided_Diffusion"
]

# Regex patterns
accuracy_pattern = re.compile(r"(CNN|CART)(?: Test)? Accuracy:\s+([0-9.]+)")
line_pattern = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)")

# Use a dict-of-dicts to merge cnn and cart into one row
data_by_model = defaultdict(dict)

for dataset_type in dataset_types:
    for size in sample_sizes:
        for model in model_folders:
            model_dir = os.path.join(base_dir, dataset_type, size, model)
            if not os.path.exists(model_dir):
                continue

            key = (dataset_type, size, model)

            for file_type in ["cnn.txt", "cart.txt"]:
                file_path = os.path.join(model_dir, file_type)
                if not os.path.exists(file_path):
                    continue

                with open(file_path, "r") as f:
                    lines = f.readlines()

                # Accuracy
                accuracy = None
                for line in lines:
                    match = accuracy_pattern.match(line.strip())
                    if match:
                        accuracy = float(match.group(2))
                        break

                prefix = file_type.replace(".txt", "").lower()  # cnn or cart
                data_by_model[key][f"{prefix}_accuracy"] = accuracy

                # Class-wise metrics
                for line in lines:
                    match = line_pattern.match(line)
                    if match:
                        cls = int(match.group(1))
                        precision = float(match.group(2))
                        recall = float(match.group(3))
                        data_by_model[key][f"{prefix}_class{cls}_precision"] = precision
                        data_by_model[key][f"{prefix}_class{cls}_recall"] = recall

# Convert to dataframe
records = []
for (dataset_type, size, model), metrics in data_by_model.items():
    record = {
        "DatasetType": dataset_type,
        "SampleSize": size,
        "ModelName": model,
        **metrics
    }
    records.append(record)

df = pd.DataFrame(records)
df.to_csv("model_accuracy_with_all_metrics.csv", index=False)
print("✅ Saved: model_accuracy_with_all_metrics.csv")
