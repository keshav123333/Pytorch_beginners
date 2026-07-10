Loss function ka kaam hota hai **model ki prediction aur actual (ground truth) ke beech kitni error hai usko measure karna**. Training ke time optimizer isi loss ko minimize karta hai.

Mathematically:

[
\theta = \theta - \eta \frac{\partial L}{\partial \theta}
]

* (\theta) = model parameters (weights)
* (\eta) = learning rate
* (L) = loss function

Jitna kam loss, utni better prediction.

---

# 1. Mean Squared Error (MSE Loss)

PyTorch:

```python
nn.MSELoss()
```

Formula:

[
L = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2
]

* (y) = actual value
* (\hat y) = predicted value

Example

Actual

```
10
```

Prediction

```
8
```

Loss

```
(10-8)^2 = 4
```

### Use

Regression

Examples

* House price prediction
* Temperature prediction
* Stock prediction

Last Layer

```
nn.Linear(...,1)
```

Activation

```
None
```

Output

```
3.52
```

Continuous value.

---

# 2. L1 Loss (MAE)

PyTorch

```python
nn.L1Loss()
```

Formula

[
L=\frac1N\sum |y-\hat y|
]

Example

Actual

```
100
```

Prediction

```
92
```

Loss

```
|100-92|=8
```

Use

Regression

Less sensitive to outliers than MSE.

Last Layer

```
Linear
```

Activation

None

---

# 3. Huber Loss (Smooth L1)

PyTorch

```python
nn.HuberLoss()
```

or

```python
nn.SmoothL1Loss()
```

Formula

For small error

[
\frac12(y-\hat y)^2
]

For large error

[
|y-\hat y|-\frac12
]

Use

Regression

Very common in Object Detection.

Example

Bounding Box Prediction

---

# 4. Binary Cross Entropy (BCELoss)

PyTorch

```python
nn.BCELoss()
```

Formula

[
L=-(y\log(p)+(1-y)\log(1-p))
]

Output probability

```
0-1
```

Last Layer

```
Sigmoid
```

Architecture

```
Linear
↓
Sigmoid
↓
BCELoss
```

Use

Binary Classification

Examples

* Cat/Dog
* Spam/Not Spam
* Disease/Healthy

---

# 5. BCEWithLogitsLoss ⭐ (Recommended)

PyTorch

```python
nn.BCEWithLogitsLoss()
```

Internally

```
Sigmoid
+
BCELoss
```

You DO NOT add sigmoid.

Architecture

```
Linear
↓
BCEWithLogitsLoss
```

NOT

```
Linear
Sigmoid
BCEWithLogitsLoss
```

because sigmoid already included.

Use

Binary Classification

---

# 6. Cross Entropy Loss ⭐⭐⭐

Most used classification loss.

PyTorch

```python
nn.CrossEntropyLoss()
```

Formula

[
L=-\log\left(\frac{e^{x_y}}{\sum e^{x_i}}\right)
]

Internally

```
Softmax
+
Negative Log Likelihood
```

Don't use Softmax.

Correct

```
Linear
↓
CrossEntropyLoss
```

Wrong

```
Linear
Softmax
CrossEntropyLoss
```

Example

3 Classes

```
Cat
Dog
Horse
```

Model Output (logits)

```
[3.2,1.1,-0.5]
```

Target

```
0
```

Use

Multi-Class Classification

Examples

* MNIST
* CIFAR10
* ImageNet
* Chest X-ray (Normal / Bacterial / Viral)

Last Layer

```python
Linear(512,3)
```

No Softmax.

---

# 7. NLL Loss

PyTorch

```python
nn.NLLLoss()
```

Needs

```
LogSoftmax
```

Architecture

```
Linear
↓
LogSoftmax
↓
NLLLoss
```

Equivalent to

```
CrossEntropyLoss
```

Nowadays mostly use CrossEntropyLoss.

---

# 8. KL Divergence Loss

PyTorch

```python
nn.KLDivLoss()
```

Formula

[
D_{KL}(P||Q)=\sum P(x)\log\frac{P(x)}{Q(x)}
]

Measures difference between two probability distributions.

Use

* Knowledge Distillation
* Variational Autoencoder (VAE)

---

# 9. Margin Ranking Loss

PyTorch

```python
nn.MarginRankingLoss()
```

Use

Ranking problems

Examples

* Search Engines
* Recommendation Systems

---

# 10. Triplet Loss

PyTorch

```python
nn.TripletMarginLoss()
```

Input

* Anchor
* Positive
* Negative

Formula

[
\max(0,d(a,p)-d(a,n)+margin)
]

Use

Face Recognition

Example

Anchor

```
Person A
```

Positive

```
Another image of A
```

Negative

```
Person B
```

---

# 11. Cosine Embedding Loss

PyTorch

```python
nn.CosineEmbeddingLoss()
```

Use

Similarity Learning

Examples

* Sentence similarity
* Image similarity

---

# 12. CTCLoss

PyTorch

```python
nn.CTCLoss()
```

Use

Sequence tasks where alignment unknown.

Examples

* OCR
* Speech Recognition

---

# 13. MultiLabelSoftMarginLoss

PyTorch

```python
nn.MultiLabelSoftMarginLoss()
```

Example

Image contains

```
Dog
Car
Tree
```

Multiple labels simultaneously.

Uses sigmoid internally for each class.

---

# 14. MultiLabelMarginLoss

Used less frequently.

Ranking based multilabel tasks.

---

# 15. PoissonNLLLoss

Use

Count prediction

Example

```
Customers
Cars
Accidents
```

---

# 16. GaussianNLLLoss

Use

Regression with uncertainty estimation.

---

# Quick Selection Table

| Problem                                | Last Layer      | Activation             | Loss                                                   |
| -------------------------------------- | --------------- | ---------------------- | ------------------------------------------------------ |
| Regression                             | `Linear(...,1)` | None                   | `MSELoss`, `L1Loss`, `HuberLoss`                       |
| Binary Classification                  | `Linear(...,1)` | None                   | `BCEWithLogitsLoss` ⭐                                  |
| Binary Classification (older approach) | `Linear(...,1)` | `Sigmoid`              | `BCELoss`                                              |
| Multi-Class Classification             | `Linear(...,C)` | None                   | `CrossEntropyLoss` ⭐                                   |
| Multi-Class (manual)                   | `Linear(...,C)` | `LogSoftmax`           | `NLLLoss`                                              |
| Multi-Label Classification             | `Linear(...,C)` | None                   | `BCEWithLogitsLoss` (one sigmoid per class internally) |
| Face Verification                      | Embedding       | Normalize (optional)   | `TripletMarginLoss`, `CosineEmbeddingLoss`             |
| Knowledge Distillation                 | Logits          | `LogSoftmax`/`Softmax` | `KLDivLoss`                                            |
| OCR / Speech                           | Sequence logits | `LogSoftmax`           | `CTCLoss`                                              |

## Important Notes

* **Regression**: Last layer linear hoti hai, activation nahi lagate.
* **Binary Classification**: Aajkal `BCEWithLogitsLoss` use karna best practice hai; model se raw logits nikalo.
* **Multi-Class Classification**: `CrossEntropyLoss` ke saath model se raw logits do. `Softmax` manually mat lagao training ke time.
* **Inference**:

  * Binary: `torch.sigmoid(logits)` se probability nikaal sakte ho.
  * Multi-class: `torch.softmax(logits, dim=1)` se probabilities aur `argmax()` se predicted class.

### Sabse zyada interview aur real projects mein use hone wale losses

1. ✅ `CrossEntropyLoss` — Image Classification, NLP Classification
2. ✅ `BCEWithLogitsLoss` — Binary aur Multi-label Classification
3. ✅ `MSELoss` — Regression
4. ✅ `L1Loss` / `HuberLoss` — Robust Regression
5. ✅ `TripletMarginLoss` — Face Recognition / Metric Learning
6. ✅ `CTCLoss` — OCR aur Speech Recognition

Ye 6 losses samajh lene se PyTorch ke adhiktar practical deep learning projects cover ho jaate hain.
