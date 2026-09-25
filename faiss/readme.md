import faiss
import numpy as np

embeddings = np.asarray(embeddings, dtype="float32")

d = embeddings.shape[1]

# IVF clusters
nlist = 4096

quantizer = faiss.IndexFlatIP(d)

index = faiss.IndexIVFFlat(
    quantizer,
    d,
    nlist,
    faiss.METRIC_INNER_PRODUCT
)

# Train on sample, NOT all 1 crore
sample_size = min(300_000, len(embeddings))

sample_idx = np.random.choice(
    len(embeddings),
    sample_size,
    replace=False
)

index.train(embeddings[sample_idx])

# Add 1 crore vectors in batches
batch_size = 100_000

for start in range(0, len(embeddings), batch_size):
    end = min(start + batch_size, len(embeddings))

    index.add(embeddings[start:end])

    print(f"Added {end:,}/{len(embeddings):,}")



index.nprobe = 20

And S1:

scores, indices = index.search(
    s1_embeddings,
    100
)
