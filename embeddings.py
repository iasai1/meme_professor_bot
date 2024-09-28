import numpy as np
from PIL import Image
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

class Embeddings:

    def __init__(self, host="localhost", port="19530"):
        # init model and processor
        model_ckpt = "jayantapaul888/vit-base-patch16-224-finetuned-memes-v2"
        self.processor = AutoImageProcessor.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt)
        self.threshold = 0.99

        # init db (milvus)
        connections.connect("default", host=host, port=port)

        # define schema for the embeddings collection
        embedding_fields = [
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="user_id", dtype=DataType.INT64),
            FieldSchema(name="message_id", dtype=DataType.INT64),
            FieldSchema(name="date", dtype=DataType.INT64)
        ]
        embedding_schema = CollectionSchema(embedding_fields, "Image embeddings with metadata")
        self.embeddings_collection = Collection("embeddings", schema=embedding_schema)

        # define schema for the bas studnets leaderboard collection
        leaderboard_fields = [
            FieldSchema(name="user_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="duplicate_count", dtype=DataType.INT64)
        ]
        leaderboard_schema = CollectionSchema(leaderboard_fields, "User leaderboard")
        self.leaderboard_collection = Collection("bad_students", schema=leaderboard_schema)


    def process_picture(self, image_path, user_id, message_id, datetime):
        """
        This function processes a new image and checks if it's a duplicate
        If image is not a duplicate, it will be added to the database
        If image is a duplicate, leaderboard will be updated

        Returns:
            int, int: user_id and target_message_id. if target message is -1 - the image was not a duplicate
        """
        new_embedding = self.get_image_embedding(image_path)
        
        # query db
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = self.embeddings_collection.search(
            [new_embedding],
            "embedding",
            param=search_params,
            limit=3, # grab top 3 just in case
            output_fields=["message_id"]
        )

        # process results
        duplicate_found = False
        target_message_id = -1
        for i, result in enumerate(results[0]):
            similarity = 1 - result.distances[i]
            if similarity > self.threshold:
                target_message_id = result.entity.get('message_id')
                duplicate_found = True
                print(f"Duplicate found with similarity: {similarity} matching message_id: {target_message_id}")
                self.update_leaderboard(user_id)
                break
                
        if not duplicate_found:
            self.insert_embedding(new_embedding, user_id, message_id, datetime)
        
        return user_id, target_message_id

    def get_image_embedding(self, image_path):
        """Process a single image and return its embedding."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_features = outputs.last_hidden_state[:, 0, :]
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten()  # Flatten to 1D array
    
    def insert_embedding(self, embedding, user_id, message_id, datetime):
        """Insert an embedding and its metadata into the embeddings collection."""
        entities = [
            [embedding],
            [user_id],
            [message_id],
            [datetime]
        ]
        self.embeddings_collection.insert(entities)
        self.embeddings_collection.flush()
        
    def update_leaderboard(self, user_id):
        """Update the leaderboard: increment duplicate count for the user, or add them with count 1 if not present."""
        results = self.leaderboard_collection.query(
            expr=f"user_id == {user_id}",
            output_fields=["user_id", "duplicate_count"]
        )
        
        entities = []
        if results:
            entities = [
                [user_id],
                [results[0]['duplicate_count'] + 1]
            ]
        else:
            entities = [
                [user_id],
                [1]
            ]
        
        self.leaderboard_collection.insert(entities)
        self.leaderboard_collection.flush()

    def get_leaderboard(self):
        """Retrieve the leaderboard sorted by duplicate count."""
        results = self.leaderboard_collection.query(expr="duplicate_count > 0", output_fields=["user_id", "duplicate_count"])
        sorted_leaderboard = sorted(results, key=lambda x: x["duplicate_count"], reverse=True)
        return sorted_leaderboard