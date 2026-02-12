#!/usr/bin/env python3
"""
Knowledge Graph Builder

Transforms a list of reading materials into a 2D-projected JSON file for the frontend.
Handles embedding generation, caching, and dimensionality reduction using t-SNE.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
READING_LIST_PATH = PROJECT_ROOT / "data" / "reading_list.json"
VECTOR_CACHE_PATH = PROJECT_ROOT / "data" / "vector_cache.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "knowledge_graph.json"

# Model configuration
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
TSNE_RANDOM_STATE = 42
NUM_CLUSTERS = 8  # Number of clusters for K-means


def load_json(path: Path, default: Any = None) -> Any:
    """Load JSON file, return default if file doesn't exist."""
    if not path.exists():
        return default if default is not None else {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_data() -> tuple[List[Dict[str, str]], Dict[str, List[float]]]:
    """Load reading list and vector cache."""
    print("Loading data sources...")
    reading_list = load_json(READING_LIST_PATH, default=[])
    vector_cache = load_json(VECTOR_CACHE_PATH, default={})

    print(f"  - Loaded {len(reading_list)} items from reading list")
    print(f"  - Loaded {len(vector_cache)} cached vectors")

    return reading_list, vector_cache


def generate_embeddings(
    reading_list: List[Dict[str, str]], vector_cache: Dict[str, List[float]]
) -> tuple[Dict[str, List[float]], bool]:
    """
    Generate embeddings for items not in cache.
    Returns updated cache and a flag indicating if new embeddings were generated.
    """
    model = None
    cache_updated = False

    for item in reading_list:
        item_id = item["id"]

        if item_id in vector_cache:
            continue

        # Lazy load model only when needed
        if model is None:
            print("Loading embedding model...")
            model = SentenceTransformer(EMBEDDING_MODEL)

        # Generate embedding
        text = f"{item['title']} {item['summary']}"
        embedding = model.encode(text, convert_to_numpy=True)
        vector_cache[item_id] = embedding.tolist()
        cache_updated = True

        print(f"  - Generated embedding for: {item['title']}")

    if not cache_updated:
        print("  - All embeddings already cached")

    return vector_cache, cache_updated


def project_to_2d(
    reading_list: List[Dict[str, str]], vector_cache: Dict[str, List[float]]
) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    """Project high-dimensional vectors to 2D using t-SNE."""
    print("Projecting vectors to 2D...")

    # Collect vectors in the same order as reading list
    vectors = []
    for item in reading_list:
        vector = vector_cache.get(item["id"])
        if vector is None:
            raise ValueError(f"Missing vector for item: {item['id']}")
        vectors.append(vector)

    # Convert to numpy array
    vectors_array = np.array(vectors)

    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        random_state=TSNE_RANDOM_STATE,
        metric="cosine",
        init="random",
        perplexity=min(30, len(vectors) - 1),  # Ensure perplexity is valid
    )

    projections = tsne.fit_transform(vectors_array)

    # Map 2D coordinates back to items
    result = []
    for item, (x, y) in zip(reading_list, projections):
        result.append(
            {
                "id": item["id"],
                "title": item["title"],
                "link": item["link"],
                "pos": [float(x), float(y)],
            }
        )

    print(f"  - Projected {len(result)} items to 2D space")

    return projections, result


def cluster_embeddings(
    projections: np.ndarray, graph_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Cluster 2D projections using K-means and add cluster info to graph data.
    """
    print("Clustering embeddings...")

    # Determine optimal number of clusters (use min of NUM_CLUSTERS or n_samples/3)
    n_samples = len(projections)
    n_clusters = min(NUM_CLUSTERS, max(2, n_samples // 3))

    # Run K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=TSNE_RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(projections)

    # Add cluster information to graph data
    for i, data_point in enumerate(graph_data):
        data_point["cluster"] = int(cluster_labels[i])

    print(f"  - Created {n_clusters} clusters")
    print(
        f"  - Cluster distribution: {np.bincount(cluster_labels).tolist()}"
    )

    return graph_data


def main():
    """Main execution flow."""
    print("=" * 60)
    print("Knowledge Graph Builder")
    print("=" * 60)

    # Step 1: Load data
    reading_list, vector_cache = load_data()

    if not reading_list:
        print("ERROR: Reading list is empty!")
        return

    # Step 2: Generate embeddings with caching
    print("\nGenerating embeddings...")
    vector_cache, cache_updated = generate_embeddings(reading_list, vector_cache)

    # Step 3: Save cache if updated
    if cache_updated:
        print("\nSaving updated cache...")
        save_json(VECTOR_CACHE_PATH, vector_cache)
        print(f"  - Cache saved to: {VECTOR_CACHE_PATH}")

    # Step 4: Project to 2D
    print("\nProjecting to 2D space...")
    projections, graph_data = project_to_2d(reading_list, vector_cache)

    # Step 5: Cluster embeddings
    print("\nClustering embeddings...")
    graph_data = cluster_embeddings(projections, graph_data)

    # Step 6: Generate output
    print("\nGenerating output...")
    save_json(OUTPUT_PATH, graph_data)
    print(f"  - Output saved to: {OUTPUT_PATH}")

    print("\n" + "=" * 60)
    print("Build complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
