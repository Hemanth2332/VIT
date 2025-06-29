from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import torch
from tqdm import tqdm
from model import VIT


DEVICE= "cuda" if torch.cuda.is_available() else "cpu"
LOAD = True
EPOCHS = 50
MODEL_PATH = "model_13.pth"


transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


dataset = ImageFolder(root="./data/caltech256", transform=transform)

train_size = int(len(dataset)*0.8)
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
classes = dataset.classes

custom_config = {
    "img_size": 224,
    "patch_size": 16,
    "in_chans": 3,
    "n_classes": len(classes),
    "embed_dim": 384,
    "depth": 8,
    "n_heads": 6,
    "qkv_bias": True,
    "mlp_ratio": 4.0,
}


model = VIT(**custom_config).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
criterion = torch.nn.CrossEntropyLoss()

classes = dataset.classes

if LOAD:
    checkpoint = torch.load()
    model.load_state_dict(checkpoint['model_weights'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    current_epoch = checkpoint['epoch']
    print(f"Model loaded from epoch {current_epoch}")

else:
    current_epoch = 0
    print("Starting training from scratch")



for epoch in range(current_epoch, EPOCHS):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for x, y in tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]"):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        
        output = model(x)
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        _, preds = torch.max(output, dim=1)
        train_acc += (preds == y).sum().item()

    train_loss /= len(train_dataloader.dataset)
    train_acc /= len(train_dataloader.dataset)
    tqdm.write(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # ----- Evaluation -----
    model.eval()
    test_loss = 0.0
    test_acc = 0.0

    with torch.no_grad():
        for x, y in tqdm(test_dataloader, desc=f"Epoch {epoch+1} [Eval]"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            output = model(x)
            loss = criterion(output, y)

            test_loss += loss.item() * x.size(0)
            _, preds = torch.max(output, dim=1)
            test_acc += (preds == y).sum().item()

    test_loss /= len(test_dataloader.dataset)
    test_acc /= len(test_dataloader.dataset)
    tqdm.write(f"Epoch {epoch+1} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    tqdm.write("--------------------------------------------------------------------------")

    checkpoint = {
        "optimizer" : optimizer.state_dict(),
        "model_weights": model.state_dict(),
        "epoch": epoch
    }

    torch.save(checkpoint, f"./models/model_food101_{epoch+1}.pth")