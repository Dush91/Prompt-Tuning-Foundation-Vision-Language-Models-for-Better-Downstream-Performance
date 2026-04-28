import os
import csv
import torch
import clip
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader

print("SCRIPT STARTED")

DATA_DIR = "./data"
RESULTS_DIR = "./results"
CHECKPOINT_DIR = "./checkpoints"

BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
N_CTX = 16
SEED = 42
NUM_WORKERS = 0

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.float()

for param in clip_model.parameters():
    param.requires_grad = False

# =====================
# LOAD OXFORD PETS
# =====================
train_dataset = datasets.OxfordIIITPet(
    root=DATA_DIR,
    split="trainval",
    download=True,
    transform=preprocess
)

test_dataset = datasets.OxfordIIITPet(
    root=DATA_DIR,
    split="test",
    download=True,
    transform=preprocess
)

classnames = train_dataset.classes
classnames = [name.replace("_", " ") for name in classnames]

print("Classes:", classnames)
print("Train images:", len(train_dataset))
print("Test images:", len(test_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# =====================
# TEXT ENCODER
# =====================
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = torch.float32

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        x = self.ln_final(x).type(self.dtype)

        eos_positions = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eos_positions] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=16):
        super().__init__()

        self.classnames = classnames
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx

        ctx_dim = clip_model.ln_final.weight.shape[0]

        ctx_vectors = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        prompts = [
            "X " * n_ctx + f"a photo of a {name}, a type of pet."
            for name in classnames
        ]

        tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in prompts]
        ).to(device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).float()

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.tokenized_prompts = tokenized_prompts

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = torch.cat(
            [
                self.token_prefix,
                ctx,
                self.token_suffix
            ],
            dim=1
        )

        return prompts


text_encoder = TextEncoder(clip_model).to(device)
prompt_learner = PromptLearner(classnames, clip_model, N_CTX).to(device)

optimizer = optim.AdamW(
    prompt_learner.parameters(),
    lr=LR,
    weight_decay=0.0001
)

criterion = nn.CrossEntropyLoss()

train_losses = []
train_accs = []

# =====================
# TRAINING
# =====================
for epoch in range(EPOCHS):
    prompt_learner.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(images).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = prompt_learner()

        text_features = text_encoder(
            prompts,
            prompt_learner.tokenized_prompts
        )

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.t()
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total

    train_losses.append(avg_loss)
    train_accs.append(train_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} Accuracy: {train_acc:.2f}%")

# =====================
# TESTING
# =====================
prompt_learner.eval()

correct = 0
total = 0

with torch.no_grad():
    prompts = prompt_learner()

    text_features = text_encoder(
        prompts,
        prompt_learner.tokenized_prompts
    )

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        image_features = clip_model.encode_image(images).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.t()
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = 100 * correct / total

# =====================
# MEAN AND VARIANCE
# =====================
mean_acc = np.mean(train_accs)
var_acc = np.var(train_accs, ddof=1)

mean_loss = np.mean(train_losses)
var_loss = np.var(train_losses, ddof=1)

print("\n===== FINAL RESULTS =====")
print(f"Test Accuracy: {test_acc:.2f}%")

print("\n===== MEAN AND VARIANCE =====")
print(f"Mean Training Accuracy: {mean_acc:.2f}%")
print(f"Variance Training Accuracy: {var_acc:.4f}")
print(f"Mean Training Loss: {mean_loss:.4f}")
print(f"Variance Training Loss: {var_loss:.4f}")

# =====================
# SAVE CSV
# =====================
csv_path = os.path.join(RESULTS_DIR, "oxfordpets_coop_results.csv")

with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow(["Epoch", "Train Loss", "Train Accuracy"])

    for i in range(EPOCHS):
        writer.writerow([i + 1, train_losses[i], train_accs[i]])

    writer.writerow([])
    writer.writerow(["Test Accuracy", test_acc])
    writer.writerow(["Mean Training Accuracy", mean_acc])
    writer.writerow(["Variance Training Accuracy", var_acc])
    writer.writerow(["Mean Training Loss", mean_loss])
    writer.writerow(["Variance Training Loss", var_loss])

print("CSV saved at:", csv_path)

# =====================
# SAVE MODEL
# =====================
model_path = os.path.join(CHECKPOINT_DIR, "oxfordpets_coop.pth")
torch.save(prompt_learner.state_dict(), model_path)
print("Model saved at:", model_path)

# =====================
# GRAPHS
# =====================
plt.figure()
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Oxford Pets CoOp Training Loss")
plt.grid(True)

loss_graph_path = os.path.join(RESULTS_DIR, "oxfordpets_loss_graph.png")
plt.savefig(loss_graph_path)
plt.close()
print("Loss graph saved at:", loss_graph_path)

plt.figure()
plt.plot(range(1, EPOCHS + 1), train_accs, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Oxford Pets CoOp Training Accuracy")
plt.grid(True)

accuracy_graph_path = os.path.join(RESULTS_DIR, "oxfordpets_accuracy_graph.png")
plt.savefig(accuracy_graph_path)
plt.close()
print("Accuracy graph saved at:", accuracy_graph_path)