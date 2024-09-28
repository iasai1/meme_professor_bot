import os
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Define the model checkpoint
model_ckpt = "jayantapaul888/vit-base-patch16-224-finetuned-memes-v2"

# Load the model and processor
processor = AutoImageProcessor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

# Directory containing your images
image_dir = "C:\\Users\\iasai\\Downloads\\Telegram Desktop\\ChatExport_2024-09-27\\photos\\"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

# Path to save/load the embeddings
embeddings_file = "embeddings.npy"

# Check if the embeddings file exists
if os.path.exists(embeddings_file):
    print(f"Loading embeddings from {embeddings_file}")
    embeddings_np = np.load(embeddings_file)
else:
    print(f"Computing embeddings and saving to {embeddings_file}")
    # Process and encode all images
    embeddings = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_features = model(**inputs).last_hidden_state[:, 0, :]  # Example processing step
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        embeddings.append(image_features.cpu().numpy())
        print(str(img_path))
    
    # Convert embeddings list to a numpy array
    embeddings_np = np.vstack(embeddings)
    
    # Save the embeddings to a file
    np.save(embeddings_file, embeddings_np)
    print(f"Embeddings saved to {embeddings_file}")

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings_np)

# Set a threshold for similarity to identify duplicates
threshold = 0.99

# Find pairs of images with similarity greater than the threshold
duplicates = []
for i in range(len(similarity_matrix)):
    for j in range(i + 1, len(similarity_matrix)):
        if similarity_matrix[i, j] > threshold:
            duplicates.append((image_paths[i], image_paths[j]))

# Output duplicate pairs
for dup in duplicates:
    print(f"Duplicate found: {dup[0]} and {dup[1]}")   

print(len(duplicates))