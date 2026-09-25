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

```
emb = F.normalize(emb, p=2, dim=1)

print(emb.shape)
