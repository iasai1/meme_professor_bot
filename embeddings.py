import json
import pprint
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

        # a flag to keep state of embeddings collection in memory
        self.loaded = False

        # init db (milvus)
        connections.connect("default", host=host, port=port)

        # define schema for the embeddings collection
        embedding_fields = [
            FieldSchema(name="primary_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="user_id", dtype=DataType.INT64),
            FieldSchema(name="message_id", dtype=DataType.INT64),
            FieldSchema(name="date", dtype=DataType.INT64)
        ]
        embedding_schema = CollectionSchema(embedding_fields, "Image embeddings with metadata")
        self.embeddings_collection = Collection("embeddings", schema=embedding_schema)

        # index embedding vector
        if not self.embeddings_collection.has_index(index_name="idx_embeddings"):
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "IP",
                "params": {"nlist": 100}
            }
            self.embeddings_collection.create_index("embedding", index_params, index_name="idx_embeddings")

        # define schema for the bad studnets leaderboard collection
        leaderboard_fields = [
            FieldSchema(name="user_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="duplicate_count", dtype=DataType.INT64),
            FieldSchema(name="dummy_vector", dtype=DataType.FLOAT_VECTOR, dim=1)
        ]
        leaderboard_schema = CollectionSchema(leaderboard_fields, "User leaderboard")
        self.leaderboard_collection = Collection("bad_students", schema=leaderboard_schema)
        if not self.leaderboard_collection.has_index(index_name="idx_dummy"):
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 1}
            }
            self.leaderboard_collection.create_index("dummy_vector", index_params, index_name="idx_dummy")


    def process_picture(self, image_path, user_id, message_id, datetime, keep_loaded=False, manual_flush=False):
        if not self.loaded:
            self.embeddings_collection.load()
            self.loaded = True
            print("Loaded collection to memory")

        r_user_id, r_message_id = self._process_picture(image_path, user_id, message_id, datetime, False) 

        if not keep_loaded:
            self.unload_db()

        return r_user_id, r_message_id
    
    def unload_db(self):
        self.embeddings_collection.release()
        self.loaded = False
        print("Unloaded collection from memory")


    def _process_picture(self, image_path, user_id, message_id, datetime, manual_flush):
        """
        This function processes a new image and checks if it's a duplicate
        If image is not a duplicate, it will be added to the database
        If image is a duplicate, leaderboard will be updated

        Returns:
            int, int: user_id and target_message_id. if target message is -1 - the image was not a duplicate
        """
        new_embedding = self.get_image_embedding(image_path)
        print(image_path)
        
        duplicate_found = False
        target_message_id = -1
    
        # query db
        search_params = {"metric_type": "IP", "params": {"nprobe": 100}}
        results = self.embeddings_collection.search(
            [new_embedding],
            "embedding",
            param=search_params,
            limit=3, # grab top 3 just in case
            output_fields=["message_id"]
        )

        for result in results:
            for i, distance in enumerate(result.distances):
                if distance > self.threshold:
                    message_id = result[i].entity.get('message_id')
                    print(f"Duplicate found with similarity: {distance}")
                    print(f"Matching message_id: {message_id}")
                    self.update_leaderboard(user_id)
                    duplicate_found = True
                    break
                
        if not duplicate_found:
            self.insert_embedding(new_embedding, user_id, message_id, datetime, manual_flush)
        
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
    
    
    def insert_embedding(self, embedding, user_id, message_id, datetime, manual_flush=False):
        """Insert an embedding and its metadata into the embeddings collection."""
        entities = [
            [embedding],
            [user_id],
            [message_id],
            [datetime]
        ]
        self.embeddings_collection.insert(entities)
        if manual_flush:
            self.embeddings_collection.flush()

        
    def update_leaderboard(self, user_id):
        """Update the leaderboard: increment duplicate count for the user, or add them with count 1 if not present."""
        self.leaderboard_collection.load()
        results = self.leaderboard_collection.query(
            expr=f"user_id == {user_id}",
            output_fields=["user_id", "duplicate_count"]
        )
        
        entities = []
        if results:
            entities = [
                [user_id],
                [results[0]['duplicate_count'] + 1],
                [[0.0]]
            ]
        else:
            entities = [
                [user_id],
                [1],
                [[0.0]]
            ]
        
        self.leaderboard_collection.insert(entities)
        self.leaderboard_collection.flush()
        self.leaderboard_collection.release()


    def get_leaderboard(self):
        """Retrieve the leaderboard sorted by duplicate count."""
        self.leaderboard_collection.load()
        results = self.leaderboard_collection.query(expr="duplicate_count > 0", output_fields=["user_id", "duplicate_count"])
        sorted_leaderboard = sorted(results, key=lambda x: x["duplicate_count"], reverse=True)
        self.leaderboard_collection.release()
        return sorted_leaderboard