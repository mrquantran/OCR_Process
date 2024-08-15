import json
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import easyocr
from usearch.index import Index


def create_global_json_path(root_dir):
    global_json_path = {}
    index = 0
    for group in os.listdir(root_dir):
        group_path = os.path.join(root_dir, group)
        if os.path.isdir(group_path):
            for video in os.listdir(group_path):
                video_path = os.path.join(group_path, video)
                if os.path.isdir(video_path):
                    for image in os.listdir(video_path):
                        if image.endswith(".webp"):
                            global_json_path[index] = (
                                f"{group}/{video}/{image.split('.')[0]}"
                            )
                            index += 1

    with open("global_json_path.json", "w") as f:
        json.dump(global_json_path, f, indent=2)

    return global_json_path


def prepare_ocr_text(ocr_list):
    filtered_list = [item.strip() for item in ocr_list if item.strip()]
    text = ". ".join(filtered_list)
    if not text.endswith("."):
        text += "."
    return text


def perform_ocr(image_path, reader):
    result = reader.readtext(image_path)
    return prepare_ocr_text([text for _, text, _ in result])


def create_embedding_index(global_json_path, root_dir, model, reader):
    index = Index(ndim=768, metric="cosine")  # 768 is the dimension of SBERT embeddings
    for i, relative_path in tqdm(
        global_json_path.items(), desc="Creating embeddings and building index"
    ):
        full_path = os.path.join(root_dir, *relative_path.split("/")) + ".webp"
        ocr_text = perform_ocr(full_path, reader)
        embedding = model.encode(ocr_text)
        index.add(int(i), embedding)

    index.save("embedding_matrix.bin")
    return index


# Initialization
root_dir = "images"
model = SentenceTransformer("keepitreal/vietnamese-sbert")
reader = easyocr.Reader(["vi"])

# Create global JSON path
global_json_path = create_global_json_path(root_dir)

# Create embedding index
usearch_index = create_embedding_index(global_json_path, root_dir, model, reader)

print(f"Global JSON path created with {len(global_json_path)} entries")
print(f"Embedding index created and saved as 'embedding_matrix.bin'")
