# Pytorch_beginners


trainable and non trainable parm ka benefit 
Bhai PyTorch me confusion isliye hota hai kyunki **3 types ki cheeze hoti hain model ke andar**:

1️⃣ **Trainable parameters**
2️⃣ **Non-trainable tensors (buffers)**
3️⃣ **Temporary tensors (forward ke andar)**

Agar ye 3 samajh gaye to kabhi confusion nahi hoga.

---

# 1️⃣ Trainable cheeze kaise banti hain

Jo cheez **`nn.Parameter`** hoti hai ya kisi layer ke weight/bias hote hain, wo **automatically trainable** ho jati hai.

Example:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(4,3)   # weight + bias trainable

model = Model()

for name, param in model.named_parameters():
    print(name, param.requires_grad)
```

Output:

```
linear.weight True
linear.bias True
```

`requires_grad=True` matlab **trainable**.

---

# 2️⃣ Agar kisi cheez ko trainable banana ho

```python
self.w = nn.Parameter(torch.randn(5,5))
```

Ye **trainable parameter ban jayega**.

Example:

```python
self.scale = nn.Parameter(torch.tensor(1.0))
```

---

# 3️⃣ Agar kisi tensor ko trainable nahi banana

Option 1️⃣ **buffer**

```python
self.register_buffer("mask", torch.ones(5,5))
```

Ye:

```
model ka part
but trainable nahi
```

---

Option 2️⃣ **requires_grad=False**

```python
w = torch.randn(5,5, requires_grad=False)
```

---

Option 3️⃣ **no_grad block**

Agar forward me calculation hai jo train nahi hona chahiye:

```python
with torch.no_grad():
    x = x + some_tensor
```

---

# 4️⃣ Temporary tensor (sabse common)

Agar tum forward me kuch add karo:

```python
x = x + 5
```

ya

```python
x = x + torch.ones(10)
```

Ye **trainable parameter nahi banega**.

Kyuki:

```
wo model attribute nahi hai
```

Example:

```python
def forward(self,x):

    x = self.linear(x)

    noise = torch.randn_like(x)

    x = x + noise

    return x
```

`noise` trainable nahi hai.

---

# 5️⃣ Kaunsi cheeze optimizer train karta hai

Optimizer sirf ye train karta hai:

```python
model.parameters()
```

Example:

```python
optimizer = torch.optim.Adam(model.parameters())
```

Isme sirf:

```
nn.Parameter
```

hi aate hain.

---

# 6️⃣ Quick cheat rule (important)

| Case                         | Trainable |
| ---------------------------- | --------- |
| layer weights                | ✔         |
| `nn.Parameter()`             | ✔         |
| `register_buffer()`          | ❌         |
| forward ke temporary tensors | ❌         |
| `requires_grad=False`        | ❌         |

---

# 7️⃣ Real example

```python
class Test(nn.Module):

    def __init__(self):
        super().__init__()

        self.w = nn.Parameter(torch.randn(3,3))   # trainable

        self.register_buffer(
            "mask",
            torch.ones(3,3)
        )   # not trainable

    def forward(self,x):

        x = x @ self.w
        x = x + self.mask

        return x
```

Trainable:

```
w
```

Not trainable:

```
mask
```

---

# 8️⃣ Sab check kaise kare

```python
for name, param in model.named_parameters():
    print(name)
```

Aur buffers check:

```python
for name, buf in model.named_buffers():
    print(name)
```

---

✅ **Ek line rule (sabse important)**

```
nn.Parameter → trainable
register_buffer → model ka part but trainable nahi
forward ke tensors → temporary
```

---

Agar chaho to main tumhe **PyTorch autograd ka ek chota sa diagram + example** dikha sakta hoon jisse **exact samajh aa jayega kaun tensor gradient le raha hai aur kaun nahi** — ye concept LLM code samajhne me bahut kaam aata hai.
