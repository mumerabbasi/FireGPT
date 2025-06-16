from pathlib import Path
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer(
    str(Path("models/minilm")),
    trust_remote_code=False,
    local_files_only=True
)

client = PersistentClient(path="data/db/chroma")
coll = client.get_collection("fire_docs")
qvec = embedder.encode("water drop altitude", normalize_embeddings=True).tolist()
hits = coll.query(query_embeddings=[qvec], n_results=3)

for summ, meta in zip(hits["documents"][0], hits["metadatas"][0]):
    print(f"{meta['pdf']} pages {meta['pages']}")
    print("→", summ[:120], "…\n")
