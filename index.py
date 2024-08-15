import json
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import easyocr
from usearch.index import Index as UsearchIndex
import faiss


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


def create_embedding_matrix(global_json_path, root_dir, model, reader):
    embedding_matrix = []
    ocr_results = {}
    for index, relative_path in tqdm(
        global_json_path.items(), desc="Creating embeddings and OCR results"
    ):
        full_path = os.path.join(root_dir, *relative_path.split("/")) + ".webp"
        ocr_text = perform_ocr(full_path, reader)
        ocr_results[index] = ocr_text
        embedding = model.encode(ocr_text)
        embedding_matrix.append(embedding)

    # Save OCR results to a JSON file
    with open("ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, ensure_ascii=False, indent=2)

    print(f"OCR results saved to ocr_results.json")

    return np.array(embedding_matrix)


def build_usearch_index(embeddings):
    dimension = embeddings.shape[1]
    usearch_index = UsearchIndex(ndim=dimension, metric="cosine")
    for i, embedding in enumerate(tqdm(embeddings, desc="Building USearch index")):
        usearch_index.add(i, embedding)
    return usearch_index


def save_usearch_index(index, file_path):
    index.save(file_path)
    print(f"USearch index saved to {file_path}")


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)
    return faiss_index


def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)
    print(f"FAISS index saved to {file_path}")


# Initialization
root_dir = "images"
model = SentenceTransformer("keepitreal/vietnamese-sbert")
reader = easyocr.Reader(["vi"])

# Create global JSON path
global_json_path = create_global_json_path(root_dir)

# Create embedding matrix and OCR results
embedding_matrix = create_embedding_matrix(global_json_path, root_dir, model, reader)

# Build and save USearch index
usearch_index = build_usearch_index(embedding_matrix)
save_usearch_index(usearch_index, "usearch_index.bin")

# Build and save FAISS index
faiss_index = build_faiss_index(embedding_matrix)
save_faiss_index(faiss_index, "faiss_index.bin")

print(f"Global JSON path created with {len(global_json_path)} entries")
print(f"Embedding matrix created with shape: {embedding_matrix.shape}")
print("USearch and FAISS indices created and saved.")
