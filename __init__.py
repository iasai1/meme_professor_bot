import os
import random
import time

from embeddings import Embeddings

# Directory containing your images
image_dir = "C:\\Users\\iasai\\Downloads\\Telegram Desktop\\ChatExport_2024-09-27\\photos\\"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
embeddings = Embeddings()

for img_path in image_paths:
    user_id = random.randint(1, 5)
    
    # Generate random message_id (for example, a large random integer)
    message_id = random.randint(100000, 999999)  # You can adjust this range as needed
    
    # Get the current time in milliseconds
    datetime = int(time.time() * 1000)
    user_id, message_id = embeddings.process_picture(img_path, user_id, message_id, datetime, keep_loaded=True, manual_flush=False)
    if message_id > -1:
        print(f"Duplicate found for {user_id}: message id {message_id}")

embeddings.unload_db()
my_image = "C:\\Users\\iasai\\Downloads\\Telegram Desktop\\ChatExport_2024-09-27\\photos\\photo_2@09-04-2023_13-40-01copy.jpg"

# test copy
user_id = random.randint(1, 5)
message_id = 1
datetime = int(time.time() * 1000)
user_id, message_id = embeddings.process_picture(my_image, user_id, message_id, datetime, keep_loaded=False, manual_flush=True)

print("Leaderboard (User ID - Duplicate Count):")
for entry in embeddings.get_leaderboard():
    user_id = entry["user_id"]
    duplicate_count = entry["duplicate_count"]
    print(f"User ID: {user_id}, Duplicate Count: {duplicate_count}")



