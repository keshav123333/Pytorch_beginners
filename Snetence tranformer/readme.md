```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

model_name = "google/muril-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "रमेश जयपुर में रहता है"

inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True
)

with torch.no_grad():
    outputs = model(**inputs)

# Mean pooling
emb = outputs.last_hidden_state.mean(dim=1)

# Normalize for cosine similarity
 
emb = F.normalize(emb, p=2, dim=1)

print(emb.shape)

```


````python
emb = model.encode(
    candidates["text"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)

toh embeddings ko disk par save kar do. Next time model se dobara encode karne ki zarurat nahi padegi.

Best simple way: .npy
import numpy as np

np.save("candidate_embeddings.npy", emb)

Baad mein:

emb = np.load("candidate_embeddings.npy")
````

faiss ka retrive karke karne ke liye hai
````python
import faiss

# 1. CPU index banao
index = faiss.IndexFlatIP(emb.shape[1])

# 2. CPU index ko GPU par transfer karo
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

# 3. Embeddings GPU index mein add karo
gpu_index.add(emb)

# 4. Query embedding
query_emb = model.encode(
    queries,
    normalize_embeddings=True
)

# 5. GPU FAISS search
scores, indices = gpu_index.search(query_emb, k=10)
````
