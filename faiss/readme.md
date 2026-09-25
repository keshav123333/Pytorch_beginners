```python
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
````

# yaha pe jab maan le tujhe emb tune npy ile mein daal di ad bahut bada hai tera emb toh tu aise kar sakta 
```python
import faiss
import numpy as np


# ============================================================
# 1. LOAD S2 + S3 EMBEDDINGS
# ============================================================
# mmap_mode="r":
# poora 1 crore embeddings ek saath RAM mein load nahi honge.
#
# IMPORTANT:
# emb2 aur emb3 ALREADY L2 NORMALIZED hain.
# Isliye neeche kahin bhi normalize_L2() nahi karna hai.

emb2 = np.load(
    "/kaggle/input/notebooks/keshavrai2004/amazonml/s2_emb.npy",
    mmap_mode="r"
)

emb3 = np.load(
    "/kaggle/input/notebooks/keshavrai2004/amazonml/s3_emb.npy",
    mmap_mode="r"
)

print("S2 shape:", emb2.shape)
print("S3 shape:", emb3.shape)


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

dimension = emb2.shape[1]

s2_len = len(emb2)
s3_len = len(emb3)

print("S2 vectors:", s2_len)
print("S3 vectors:", s3_len)
print("Total vectors:", s2_len + s3_len)
print("Dimension:", dimension)


# ============================================================
# 3. CREATE IVF INDEX
# ============================================================
#
# We want:
#
# S2:
#   0 ... s2_len-1
#
# S3:
#   s2_len ... s2_len+s3_len-1
#
# Since we are manually assigning these IDs using
# add_with_ids(), we wrap IVF inside IndexIDMap2.
#
# ============================================================

nlist = 4096

quantizer = faiss.IndexFlatIP(dimension)

ivf_index = faiss.IndexIVFFlat(
    quantizer,
    dimension,
    nlist,
    faiss.METRIC_INNER_PRODUCT
)

index = faiss.IndexIDMap2(ivf_index)


# ============================================================
# 4. TRAIN IVF
# ============================================================
#
# We DO NOT train on all 1 crore embeddings.
#
# We take 300k samples:
#   150k from S2
#   150k from S3
#
# IMPORTANT:
# Embeddings are already normalized.
# So NO normalize_L2() here.
#
# ============================================================

sample_size = min(
    300_000,
    s2_len + s3_len
)

rng = np.random.default_rng(42)


# -------------------------
# Sample from S2
# -------------------------

sample_s2 = min(
    sample_size // 2,
    s2_len
)


# -------------------------
# Sample from S3
# -------------------------

sample_s3 = min(
    sample_size - sample_s2,
    s3_len
)


idx2 = rng.choice(
    s2_len,
    size=sample_s2,
    replace=False
)

idx3 = rng.choice(
    s3_len,
    size=sample_s3,
    replace=False
)


# ============================================================
# Create temporary training data
# ============================================================

train_data = np.concatenate(
    [
        np.asarray(
            emb2[idx2],
            dtype="float32"
        ),

        np.asarray(
            emb3[idx3],
            dtype="float32"
        )
    ],
    axis=0
)


print("\nTraining data shape:", train_data.shape)

print("Training IVF index...")

index.train(train_data)

print("Training completed.")


# ============================================================
# Free training memory
# ============================================================

del train_data
del idx2
del idx3


# ============================================================
# 5. ADD S2
# ============================================================
#
# S2 IDs:
#
# 0
# 1
# 2
# ...
# s2_len - 1
#
# ============================================================

batch_size = 100_000

print("\n==============================")
print("Adding S2")
print("==============================")


for start in range(
    0,
    s2_len,
    batch_size
):

    end = min(
        start + batch_size,
        s2_len
    )


    # --------------------------------------------------------
    # Load only current batch
    # --------------------------------------------------------

    batch = np.asarray(
        emb2[start:end],
        dtype="float32"
    ).copy()


    # --------------------------------------------------------
    # IMPORTANT:
    # emb2 is ALREADY normalized.
    # So DO NOT call faiss.normalize_L2(batch)
    # --------------------------------------------------------


    # --------------------------------------------------------
    # S2 global IDs
    # --------------------------------------------------------

    ids = np.arange(
        start,
        end,
        dtype=np.int64
    )


    # --------------------------------------------------------
    # Add to IVF
    # --------------------------------------------------------

    index.add_with_ids(
        batch,
        ids
    )


    # --------------------------------------------------------
    # Free temporary memory
    # --------------------------------------------------------

    del batch
    del ids


    print(
        f"S2 added: {end:,}/{s2_len:,}"
    )


# ============================================================
# 6. ADD S3
# ============================================================
#
# S3 IDs start AFTER S2.
#
# Example:
#
# S2 = 5,000,000
#
# S2:
#   0
#   ...
#   4,999,999
#
# S3:
#   5,000,000
#   ...
#   9,999,999
#
# ============================================================

s3_offset = s2_len

print("\n==============================")
print("Adding S3")
print("==============================")


for start in range(
    0,
    s3_len,
    batch_size
):

    end = min(
        start + batch_size,
        s3_len
    )


    # --------------------------------------------------------
    # Load current S3 batch only
    # --------------------------------------------------------

    batch = np.asarray(
        emb3[start:end],
        dtype="float32"
    ).copy()


    # --------------------------------------------------------
    # IMPORTANT:
    # emb3 is ALREADY normalized.
    # No normalize_L2()
    # --------------------------------------------------------


    # --------------------------------------------------------
    # GLOBAL S3 IDs
    # --------------------------------------------------------

    ids = np.arange(
        s3_offset + start,
        s3_offset + end,
        dtype=np.int64
    )


    # --------------------------------------------------------
    # Add to IVF
    # --------------------------------------------------------

    index.add_with_ids(
        batch,
        ids
    )


    # --------------------------------------------------------
    # Free temporary memory
    # --------------------------------------------------------

    del batch
    del ids


    print(
        f"S3 added: {end:,}/{s3_len:,}"
    )


# ============================================================
# 7. IVF SEARCH PARAMETERS
# ============================================================

ivf_index.nprobe = 50


print("\n==============================")
print("FAISS INDEX READY")
print("==============================")

print(
    "Total vectors:",
    index.ntotal
)

print(
    "nlist:",
    ivf_index.nlist
)

print(
    "nprobe:",
    ivf_index.nprobe
)


# ============================================================
# 8. S1 → FAISS TOP-K
# ============================================================

dic = {}

k = 15


def make_bucket(s1):

    S1 = s1.copy()


    # --------------------------------------------------------
    # Create text
    # --------------------------------------------------------

    S1["text"] = S1.apply(
        make_text,
        axis=1
    )


    # --------------------------------------------------------
    # Query embeddings
    # --------------------------------------------------------
    #
    # normalize_embeddings=True is CORRECT here.
    #
    # S1 query embeddings need to be normalized because
    # FAISS is using INNER_PRODUCT.
    #
    # Since S2/S3 embeddings are already normalized,
    # inner product = cosine similarity.
    #
    # --------------------------------------------------------

    query_emb = model.encode(
        S1["text"].tolist(),

        normalize_embeddings=True,

        show_progress_bar=True
    )


    query_emb = np.asarray(
        query_emb,
        dtype="float32"
    )


    # --------------------------------------------------------
    # FAISS SEARCH
    # --------------------------------------------------------

    scores, faiss_indices = index.search(
        query_emb,
        k
    )


    # --------------------------------------------------------
    # SAVE CANDIDATES
    # --------------------------------------------------------

    for i in range(len(S1)):

        score1 = scores[i].tolist()

        index1 = faiss_indices[i].tolist()


        dic[
            S1.iloc[i]["entity_id"]
        ] = [
            index1,
            score1
        ]


# ============================================================
# RUN
# ============================================================

make_bucket(s1)
```
