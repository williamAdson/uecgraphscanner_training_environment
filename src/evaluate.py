import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = BinaryFocalLoss()

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()} 
            logits = model(batch)

            loss = criterion(logits, batch["labels"])
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0)
    }
    return metrics, confusion_matrix(all_labels, all_preds)