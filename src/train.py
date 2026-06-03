import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from src.evaluate import evaluate, BinaryFocalLoss
from src.utils import save_checkpoint

def train_model(model, train_loader, val_loader, config, device):
    c = config['training']
    os.makedirs(config['output']['checkpoint_dir'], exist_ok=True)
    
    optimizer = AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    scheduler = OneCycleLR(optimizer, max_lr=c['lr'], steps_per_epoch=len(train_loader), epochs=c['epochs'])
    criterion = BinaryFocalLoss(alpha=c['focal_alpha'], gamma=c['focal_gamma'])

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(c['epochs']):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{c['epochs']}")
        for batch in pbar:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
                
            optimizer.zero_grad()
            logits = model(batch)
            
            loss = criterion(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_metrics, conf_mat = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Confusion Matrix:\n{conf_mat}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, os.path.join(config['output']['checkpoint_dir'], "best_model.pth"))
            print(">>> New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= c['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break