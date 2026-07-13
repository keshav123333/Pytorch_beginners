# Bootstrap Method 
```python
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )
    
    
    def bootstrap_test(
            model1,
            model2,
            test_loader,
            metric="accuracy",
            n_bootstrap=10000,
            device="cuda",
            seed=42,
            plot=True
    ):

    model1.to(device).eval()
    model2.to(device).eval()

    y_true = []
    pred1 = []
    pred2 = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            out1 = model1(images)
            out2 = model2(images)

            p1 = torch.argmax(out1, dim=1).cpu().numpy()
            p2 = torch.argmax(out2, dim=1).cpu().numpy()

            pred1.extend(p1)
            pred2.extend(p2)
            y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)

    ######################################################
    # Metric Function
    ######################################################

    def score(y, pred):

        if metric.lower() == "accuracy":
            return accuracy_score(y, pred)

        elif metric.lower() == "precision":
            return precision_score(
                y,
                pred,
                average="weighted",
                zero_division=0
            )

        elif metric.lower() == "recall":
            return recall_score(
                y,
                pred,
                average="weighted",
                zero_division=0
            )

        elif metric.lower() == "f1":
            return f1_score(
                y,
                pred,
                average="weighted",
                zero_division=0
            )

        else:
            raise ValueError(
                "metric must be accuracy / precision / recall / f1"
            )

    ######################################################
    # Original Scores
    ######################################################

    original_model1 = score(y_true, pred1)
    original_model2 = score(y_true, pred2)

    ######################################################
    # Bootstrap
    ######################################################

    rng = np.random.default_rng(seed)

    N = len(y_true)

    diff_scores = []

    for _ in range(n_bootstrap):

        idx = rng.choice(
            N,
            size=N,
            replace=True
        )

        yt = y_true[idx]

        p1 = pred1[idx]
        p2 = pred2[idx]

        s1 = score(yt, p1)
        s2 = score(yt, p2)

        diff_scores.append(s1 - s2)

    diff_scores = np.array(diff_scores)

    ######################################################
    # Statistics
    ######################################################

    mean_diff = diff_scores.mean()

    ci_low = np.percentile(diff_scores, 2.5)
    ci_high = np.percentile(diff_scores, 97.5)

    p_value = 2 * min(
        np.mean(diff_scores >= 0),
        np.mean(diff_scores <= 0)
    )

    ######################################################
    # Histogram
    ######################################################

    if plot:

        plt.figure(figsize=(8,5))
        plt.hist(diff_scores, bins=40)
        plt.axvline(mean_diff, linestyle="--")
        plt.xlabel("Metric Difference (Model1 - Model2)")
        plt.ylabel("Frequency")
        plt.title("Bootstrap Distribution")
        plt.show()

    ######################################################
    # Print Report
    ######################################################

    print("="*60)

    print(f"Metric                 : {metric}")

    print(f"Model 1 Score          : {original_model1:.4f}")

    print(f"Model 2 Score          : {original_model2:.4f}")

    print(f"Mean Difference        : {mean_diff:.4f}")

    print(f"95% Confidence Interval: ({ci_low:.4f}, {ci_high:.4f})")

    print(f"P-value                : {p_value:.6f}")

    if p_value < 0.05:
        print("\nResult : Statistically Significant")
    else:
        print("\nResult : NOT Statistically Significant")

    print("="*60)

    ######################################################
    # Return
    ######################################################

    return {
        "metric": metric,
        "model1_score": original_model1,
        "model2_score": original_model2,
        "mean_difference": mean_diff,
        "confidence_interval": (ci_low, ci_high),
        "p_value": p_value,
        "bootstrap_distribution": diff_scores
    }

````

then fir ye 

    result = bootstrap_test(
        model1=model_resnet,
        model2=model_efficientnet,
        test_loader=test_loader,
        metric="accuracy",
        n_bootstrap=10000,
        device="cuda"
    )
    
    Ya
    
    result = bootstrap_test(
        model1=model1,
        model2=model2,
        test_loader=test_loader,
        metric="f1",
        n_bootstrap=5000,
        device="cuda"
    )
