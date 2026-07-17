
Haan, tumhare logs dekh ke lag raha hai ki **overfitting ho rahi hai**.

* Train Acc = **99.8%**
* Val Acc ≈ **95%**
* Train Loss ≈ **0.004**
* Val Loss = **0.8–1.1**

Ye clear sign hai ki model training set ko memorize kar raha hai.

## 1. AdamW me weight_decay add karo (sabse pehle ye)

Abhi tumne sirf LR diya hai:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=0.05
)
```

Ya

```python
weight_decay=1e-2
```

Dono try kar sakte ho. Swin Transformer ke liye **0.05** kaafi commonly use hota hai.

---

## 2. Label Smoothing lagao (Highly Recommended)

CrossEntropyLoss ki jagah:

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

Ye transformers me bahut effective hota hai.

---

## 3. Dropout increase karo

Swin me head ke dropout ko change kar sakte ho.

```python
model = timm.create_model(
    "swin_base_patch4_window7_224",
    pretrained=True,
    num_classes=4,
    drop_rate=0.3
)
```

Ya

```python
drop_rate=0.2
```

Agar support kare to.

---

## 4. Stochastic Depth (Drop Path)

Ye Transformer models me aur bhi effective hota hai.

```python
model = timm.create_model(
    "swin_base_patch4_window7_224",
    pretrained=True,
    num_classes=4,
    drop_path_rate=0.2
)
```

Agar overfitting zyada hai to

```python
drop_path_rate=0.3
```

bhi try kar sakte ho.

---

## 5. Data Augmentation badhao

```python
RandomHorizontalFlip()
RandomRotation(15)
RandomResizedCrop(224)
ColorJitter()
RandomErasing()
```

MixUp aur CutMix bhi Swin ke saath kaafi achha perform karte hain.

---

## 6. Early Stopping

Tumhare logs me best model almost Epoch 2–4 ke aas-paas tha.

30 epochs ki zarurat nahi lag rahi.

Patience = 5 rakh do.

---

## 7. Freeze kuch layers

Agar dataset chhota hai (maan lo 5k–10k images), to pura Swin fine-tune mat karo.

Example:

```python
for param in model.parameters():
    param.requires_grad = False

for param in model.head.parameters():
    param.requires_grad = True
```

Ya last stage bhi unfreeze kar sakte ho.

---

## Mere hisaab se sabse effective combination

```python
model = timm.create_model(
    "swin_base_patch4_window7_224",
    pretrained=True,
    num_classes=4,
    drop_path_rate=0.2,
    drop_rate=0.2
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=0.05
)
```

### Ek sawal:

**Tumhare dataset me total kitni training images hain (4 classes me kitni-kitni)?** Dataset size pata chalega to bata sakta hoon ki full fine-tuning karni chahiye ya sirf last layers train karni chahiye.
