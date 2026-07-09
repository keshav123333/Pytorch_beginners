😂 Bhai confusion ho gaya. **Do model save nahi karne hote.** Maine **2 alag examples** diye the.

* `checkpoint.pth` → training ke beech (best model ya har epoch ke baad)
* `last_checkpoint.pth` → training khatam hone ke baad

**Dono me se sirf ek hi use karna hai.**

Main ab **step-by-step** batata hoon.

---

# STEP 1: Model banao

```python
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features,2)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.fc.parameters(),lr=0.001)
```

---

# STEP 2: Training Start

```python
epochs = 10

best_acc = 0
```

---

# STEP 3: Training Loop

Ye tumhare code me already hai.

```python
for epoch in range(epochs):

    train_loss, train_acc = train(model, train_loader)

    val_loss, val_acc = validate(model, val_loader)

    print(train_acc, val_acc)
```

---

# STEP 4: YAHI PAR MODEL SAVE KARNA HAI

Ye code **training loop ke andar** lagana hai.

```python
if val_acc > best_acc:

    best_acc = val_acc

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }, "checkpoint.pth")
```

Ab kya hoga?

Maan lo

```
Epoch1 Accuracy = 82%
```

save ho gaya.

Fir

```
Epoch2 Accuracy = 84%
```

purana overwrite ho jayega.

Fir

```
Epoch3 Accuracy = 81%
```

save nahi hoga.

Fir

```
Epoch4 Accuracy = 88%
```

fir overwrite.

Is tarah **sirf best model** save rahega.

---

# STEP 5: Training Complete

Ab Colab me tumhare folder me ek file hogi

```
checkpoint.pth
```

Bas.

**Aur kuch nahi.**

---

# STEP 6: Google Drive me Save (Optional)

```python
from google.colab import drive

drive.mount('/content/drive')
```

Fir

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_acc': best_acc
}, "/content/drive/MyDrive/checkpoint.pth")
```

Ya

Download

```python
from google.colab import files

files.download("checkpoint.pth")
```

---

# STEP 7: Kaggle me Upload

Upload karo

```
checkpoint.pth
```

---

# STEP 8: Kaggle me Same Model Banao

```python
model = models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features,2)

model = model.to(device)

optimizer = optim.Adam(model.fc.parameters(),lr=0.001)
```

---

# STEP 9: Load

```python
checkpoint = torch.load("checkpoint.pth", map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

start_epoch = checkpoint['epoch'] + 1

best_acc = checkpoint['best_acc']
```

Ab maan lo Colab me

```
Epoch 0
Epoch 1
Epoch 2
Epoch 3
Epoch 4
Epoch 5
```

tak train hua tha.

To

```
start_epoch = 6
```

ho jayega.

---

# STEP 10: Continue Training

```python
epochs = 15

for epoch in range(start_epoch, epochs):

    train_loss, train_acc = train(model, train_loader)

    val_loss, val_acc = validate(model, val_loader)

    print(train_acc, val_acc)
```

Output

```
Epoch 6
Epoch 7
Epoch 8
...
Epoch 14
```

Training wahi se continue hogi.

---

# Tumhare code me sirf ye 3 jagah change karni hain:

### ✅ 1. Training se pehle

```python
best_acc = 0
```

### ✅ 2. Training loop ke andar

```python
if val_acc > best_acc:
    best_acc = val_acc

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }, "checkpoint.pth")
```

### ✅ 3. Kaggle me

```python
checkpoint = torch.load("checkpoint.pth")

model.load_state_dict(checkpoint['model_state_dict'])

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

start_epoch = checkpoint['epoch'] + 1
```

---

**Mujhe lagta hai tum pehli baar checkpointing seekh rahe ho.** Agar haan, to main **tumhare isi ResNet18 pneumonia code me line-by-line bata sakta hoon ki kis line ke baad kaunsa code paste karna hai**, taaki bina confusion ke Colab → Kaggle resume training ho jaye.
