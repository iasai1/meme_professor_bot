import pprint
from PIL import Image
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import torch
import easyocr
from paddleocr import PaddleOCR
from PIL import Image


class Embeddings:

    def __init__(self, host="localhost", port="19530", threshold=0.99):

        self.device = device="cuda" if torch.cuda.is_available() else "cpu"

        # init CLIP model and processor for image and text embedding
        img_model_ckpt = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(img_model_ckpt)
        self.clip_model = CLIPModel.from_pretrained(img_model_ckpt).to(device)

        # Initialize BLIP model and processor for captioning
        txt_model_ckpt = "Salesforce/blip-image-captioning-base"
        self.txt_processor = BlipProcessor.from_pretrained(txt_model_ckpt)
        self.txt_model = BlipForConditionalGeneration.from_pretrained(txt_model_ckpt).to(device)

        # a flag to keep state of embeddings collection in memory
        self.loaded = False

        self.threshold = threshold

        # init db (milvus)
        connections.connect("default", host=host, port=port)

        # define schema for the embeddings collection
        embedding_fields = [
            FieldSchema(name="primary_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
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
                "params": {"nlist": 10}
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

        r_user_id, r_message_id = self._process_picture(image_path, user_id, message_id, datetime, manual_flush) 

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
        new_embedding = self.get_combined_image_text_embedding(image_path)
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

        pprint.pprint(results)

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
    
    def get_image_embedding(self, image):
        """Get the image embedding using CLIP."""
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize to unit length

        return image_features.cpu().numpy().flatten()  # Flatten to 1D array

    def get_text_embedding(self, description):
        """Get the text embedding using CLIP."""
        inputs = self.clip_processor(text=description, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize to unit length

        return text_features.cpu().numpy().flatten()  # Flatten to 1D array
    
    def generate_image_description(self, image):
        """Generate a description for the image using BLIP."""
        inputs = self.txt_processor(images=image, return_tensors="pt").to(self.device)
        inputs = self.txt_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.txt_model.generate(**inputs, max_length=200, num_beams=5, repetition_penalty=2.5, early_stopping=True)
            description = self.txt_processor.decode(generated_ids[0], skip_special_tokens=True)

        print(description)
        return description

    def get_combined_image_text_embedding(self, image_path):
        """Get both image and text embeddings and return them combined."""
        image = Image.open(image_path).convert("RGB")
        image_embedding = self.get_image_embedding(image)

        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en', 'ru'])  # Supports English ('en') and Russian ('ru')

        # Perform OCR on the image
        result = reader.readtext(image_path, detail=0)
        print(result)

        # Initialize PaddleOCR with language support
        ocr = PaddleOCR(use_angle_cls=True, lang='cyrillic')  # English OCR

        # Perform OCR
        result = ocr.ocr(image_path)

        # Print the recognized text
        for line in result:
            print("Detected Text:", [t[1][0] for t in line])
        # text_embedding = self.get_text_embedding(self.generate_image_description(image))
        # combined_embedding = (image_embedding + text_embedding) / 2

        # return combined_embedding
        return image_embedding
    
    
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