"""
Goal is to fine-tune a small encoder model (ideally multilingual) with a simple classification head
-> Predicts if this token is EOS or not

Caution: strong class imbalance -> find a way to remedy that"""

from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.utils.data import DataLoader
from data_procesing import EOSDataset, load_raw_data
from torch.optim import AdamW
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

model_name = "bert-base-multilingual-cased"
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# We train only the classification head (My PC is not very powerful)
for param in model.bert.parameters():
    param.requires_grad = False

raw_train = load_raw_data(split='test')
raw_val = load_raw_data(split='dev')

train_set = EOSDataset(raw_train, tokenizer)
val_set = EOSDataset(raw_val, tokenizer)

train_loader = DataLoader(train_set,batch_size=16,shuffle=True)
val_loader = DataLoader(val_set,batch_size=16,shuffle=True)


optimizer = AdamW(model.parameters(), lr=1e-3)
# LR élevé car on n'entraîne que la tête — pas de risque de catastrophic forgetting

scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

# Define our own weighted loss
n_zeros = sum((b["labels"] == 0).sum().item() for b in train_loader)
n_ones  = sum((b["labels"] == 1).sum().item() for b in train_loader)
ratio   = n_zeros / n_ones  # ~40x dans ton cas
print(f"Ratio 0/1 : {ratio:.1f}")

loss_fn = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, ratio]),
    ignore_index=-100         
)

NUM_EPOCHS = 25
pbar = tqdm(range(NUM_EPOCHS))

for epoch in pbar:

    pbar.set_description(f"Training epoch {epoch + 1}/{NUM_EPOCHS}")

    # --- Training ---
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        outputs = model(
            input_ids=      batch["input_ids"],
            attention_mask= batch["attention_mask"],
        )

        logits = outputs.logits
        labels = batch["labels"]

        # Reshape pour CrossEntropyLoss : (batch*seq_len, 2) et (batch*seq_len,)
        loss = loss_fn(
            logits.view(-1, 2),
            labels.view(-1),
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # --- Validation ---
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=      batch["input_ids"],
                attention_mask= batch["attention_mask"],
            )
            preds = outputs.logits.argmax(dim=-1)  # (batch, seq_len)

            # Flatten et filtre les -100
            for i in range(len(batch["labels"])):
                for j in range(len(batch["labels"][i])):
                    if batch["labels"][i][j] != -100:
                        all_preds.append(preds[i][j].item())
                        all_true.append(batch["labels"][i][j].item())

    f1 = f1_score(all_true, all_preds)
    scheduler.step(f1)
    print(f"Epoch {epoch+1} — loss: {total_loss/len(train_loader):.4f}  F1: {f1:.4f}")

    # Stopping criterion
    if f1 > 0.98:
        break


## Saving model
torch.save(model.classifier.state_dict(), "eos_head.pt")