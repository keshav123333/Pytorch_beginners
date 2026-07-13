 ### 1. NAdam (PyTorch)

PyTorch me directly available hai:

```python
import torch

model = MyModel()

optimizer = torch.optim.NAdam(
    model.parameters(),
    lr=0.001
)
```

Isme Adam ke saath Nesterov (look-ahead idea) use hota hai.

---

### 2. Lookahead + Adam

Original PyTorch me `torch.optim.Lookahead` built-in nahi hai. Iske liye external libraries (jaise `timm` ya `torch-optimizer`) use ki jati hain.

Example:

```python
from torch_optimizer import Lookahead

base_optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

optimizer = Lookahead(base_optimizer)
```

---
 
 
