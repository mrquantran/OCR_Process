import json
from sentence_transformers import SentenceTransformer
from usearch.index import Index


def load_global_json_path(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def search(query, model, index, top_k=20):
    query_embedding = model.encode(query)
    results = index.search(query_embedding, top_k)
    return [(int(match.key), match.distance) for match in results]


# Load the global JSON path
global_json_path = load_global_json_path("global_json_path.json")

# Load the USearch index
usearch_index = Index.restore("embedding_matrix.bin")

# Initialize the SentenceTransformer model
model = SentenceTransformer("keepitreal/vietnamese-sbert")

# Perform a search
query = "người việt nam sẽ ủng hộ lệnh cấm nạn buôn bán thịt chó mèo"
results = search(query, model, usearch_index)

print(f"Search results for query: '{query}'")
for i, (id, distance) in enumerate(results[:5]):  # Only print the top 5 results
    print(
        f"Result {i+1}: Image ID: {id}, Path: {global_json_path[str(id)]}, Similarity: {1-distance:.4f}"
    )
