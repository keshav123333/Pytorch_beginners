2. Agar model complete train ho gaya aur deploy karna hai

Ab checkpoint save karne ki bhi zarurat nahi.

Deployment ke liye sirf model ke weights save karo.

torch.save(model.state_dict(), "best_model.pth")


 
        from fastapi import FastAPI, UploadFile, File
        from PIL import Image
        import io
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        
        app = FastAPI()
        
        device = torch.device("cpu")
        
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features,2)
        
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485,0.456,0.406],
                [0.229,0.224,0.225]
            )
        ])
        
        classes = ["Normal","Pneumonia"]
        
        
        @app.post("/predict")
        async def predict(file: UploadFile = File(...)):
        
            image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        
            image = transform(image)
        
            image = image.unsqueeze(0)
        
            with torch.no_grad():
                output = model(image)
                pred = output.argmax(1).item()

    return {
        "prediction": classes[pred]
    }
