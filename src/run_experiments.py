import os, math, json, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE_DIR, 'generated_results')
os.makedirs(OUT, exist_ok=True)

device='cpu'

# Load local digits dataset (8x8 grayscale)
d = load_digits()
X = d.data.astype(np.float32) / 16.0
Y = d.target.astype(np.int64)
images = d.images.astype(np.float32) / 16.0

X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X, Y, images, test_size=0.2, random_state=SEED, stratify=Y
)

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)
img_test_t = torch.tensor(img_test)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

def eval_model(model, loader):
    model.eval()
    ys=[]; yh=[]; total_loss=0; n=0
    ce=nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb,yb in loader:
            out=model(xb)
            loss=ce(out,yb)
            total_loss += loss.item()*len(xb)
            n += len(xb)
            ys.extend(yb.numpy().tolist())
            yh.extend(out.argmax(1).numpy().tolist())
    acc=accuracy_score(ys,yh)
    p,r,f,_ = precision_recall_fscore_support(ys,yh,average='macro',zero_division=0)
    return {'loss': total_loss/n, 'acc':acc, 'precision':p,'recall':r,'f1':f, 'y_true':ys,'y_pred':yh}

def fgsm_attack(model, x, y, eps=0.15):
    x = x.clone().detach().requires_grad_(True)
    out = model(x)
    loss = nn.CrossEntropyLoss()(out, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    adv = x + eps * x.grad.sign()
    return adv.clamp(0,1).detach()

def pgd_attack(model, x, y, eps=0.2, alpha=0.04, steps=10):
    x0 = x.clone().detach()
    adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    adv = adv.clamp(0,1)
    for _ in range(steps):
        adv.requires_grad_(True)
        out = model(adv)
        loss = nn.CrossEntropyLoss()(out,y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        adv = adv.detach() + alpha * adv.grad.sign()
        delta = torch.clamp(adv - x0, min=-eps, max=eps)
        adv = (x0 + delta).clamp(0,1).detach()
    return adv

def eval_under_attack(model, X, Y, attack='fgsm', eps=0.15, alpha=0.04, steps=10, batch=256):
    model.eval()
    yh=[]; ys=[]
    for i in range(0,len(X),batch):
        xb = X[i:i+batch]
        yb = Y[i:i+batch]
        if attack=='fgsm':
            adv = fgsm_attack(model, xb, yb, eps)
        else:
            adv = pgd_attack(model, xb, yb, eps, alpha, steps)
        with torch.no_grad():
            pred = model(adv).argmax(1)
        yh.extend(pred.numpy().tolist())
        ys.extend(yb.numpy().tolist())
    acc=accuracy_score(ys,yh)
    p,r,f,_ = precision_recall_fscore_support(ys,yh,average='macro',zero_division=0)
    return {'acc':acc,'precision':p,'recall':r,'f1':f, 'y_true':ys,'y_pred':yh}

def train_model(model, train_loader, test_loader, epochs=20, adv_train=False, eps=0.15):
    opt=optim.Adam(model.parameters(), lr=0.001)
    ce=nn.CrossEntropyLoss()
    history={'train_loss':[],'val_loss':[],'val_acc':[]}
    for epoch in range(epochs):
        model.train()
        losses=[]
        for xb,yb in train_loader:
            if adv_train:
                # warmup on clean in first 2 epochs for stability
                if epoch >= 2:
                    adv = fgsm_attack(model, xb, yb, eps)
                    xb_use = torch.cat([xb, adv], dim=0)
                    yb_use = torch.cat([yb, yb], dim=0)
                else:
                    xb_use, yb_use = xb, yb
            else:
                xb_use, yb_use = xb, yb
            opt.zero_grad(set_to_none=True)
            out = model(xb_use)
            loss = ce(out, yb_use)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        metrics = eval_model(model, test_loader)
        history['train_loss'].append(float(np.mean(losses)))
        history['val_loss'].append(metrics['loss'])
        history['val_acc'].append(metrics['acc'])
    return history

# Train baseline and robust models
baseline = MLP()
robust = MLP()
base_hist = train_model(baseline, train_loader, test_loader, epochs=20, adv_train=False)
rob_hist = train_model(robust, train_loader, test_loader, epochs=20, adv_train=True, eps=0.15)

# Evaluate
clean_base = eval_model(baseline, test_loader)
clean_rob = eval_model(robust, test_loader)
fgsm_base = eval_under_attack(baseline, X_test, y_test, attack='fgsm', eps=0.15)
pgd_base = eval_under_attack(baseline, X_test, y_test, attack='pgd', eps=0.20, alpha=0.04, steps=10)
fgsm_rob = eval_under_attack(robust, X_test, y_test, attack='fgsm', eps=0.15)
pgd_rob = eval_under_attack(robust, X_test, y_test, attack='pgd', eps=0.20, alpha=0.04, steps=10)

summary = {
    'dataset': {'name': 'sklearn digits', 'samples': int(len(X)), 'classes': 10, 'image_shape': [8,8]},
    'baseline': {
        'clean_acc': clean_base['acc'], 'clean_f1': clean_base['f1'],
        'fgsm_acc': fgsm_base['acc'], 'pgd_acc': pgd_base['acc']
    },
    'robust': {
        'clean_acc': clean_rob['acc'], 'clean_f1': clean_rob['f1'],
        'fgsm_acc': fgsm_rob['acc'], 'pgd_acc': pgd_rob['acc']
    }
}
with open(os.path.join(OUT,'results.json'),'w') as f:
    json.dump(summary,f,indent=2)

# Plot loss curves
plt.figure(figsize=(7,4.2))
plt.plot(base_hist['train_loss'], label='Baseline train loss')
plt.plot(base_hist['val_loss'], label='Baseline validation loss')
plt.plot(rob_hist['train_loss'], label='Adv-trained train loss')
plt.plot(rob_hist['val_loss'], label='Adv-trained validation loss')
plt.xlabel('Epoch')
plt.ylabel('Cross-entropy loss')
plt.title('Training dynamics on handwritten digits benchmark')
plt.legend(frameon=False, ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'loss_curves.png'), dpi=220)
plt.close()

# Plot clean accuracy curves
plt.figure(figsize=(7,4.2))
plt.plot(base_hist['val_acc'], label='Baseline')
plt.plot(rob_hist['val_acc'], label='Adversarial training')
plt.xlabel('Epoch')
plt.ylabel('Validation accuracy')
plt.title('Validation accuracy during training')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'accuracy_curves.png'), dpi=220)
plt.close()

# Attack impact over epsilon
fgsm_eps = [0.0,0.05,0.10,0.15,0.20,0.25,0.30]
pgd_eps = [0.0,0.05,0.10,0.15,0.20,0.25,0.30]
fgsm_base_acc=[]; fgsm_rob_acc=[]; pgd_base_acc=[]; pgd_rob_acc=[]
for e in fgsm_eps:
    if e==0:
        fgsm_base_acc.append(clean_base['acc']); fgsm_rob_acc.append(clean_rob['acc'])
    else:
        fgsm_base_acc.append(eval_under_attack(baseline, X_test, y_test, attack='fgsm', eps=e)['acc'])
        fgsm_rob_acc.append(eval_under_attack(robust, X_test, y_test, attack='fgsm', eps=e)['acc'])
for e in pgd_eps:
    if e==0:
        pgd_base_acc.append(clean_base['acc']); pgd_rob_acc.append(clean_rob['acc'])
    else:
        a=max(0.01,e/5)
        pgd_base_acc.append(eval_under_attack(baseline, X_test, y_test, attack='pgd', eps=e, alpha=a, steps=10)['acc'])
        pgd_rob_acc.append(eval_under_attack(robust, X_test, y_test, attack='pgd', eps=e, alpha=a, steps=10)['acc'])

plt.figure(figsize=(7,4.2))
plt.plot(fgsm_eps, fgsm_base_acc, marker='o', label='FGSM baseline')
plt.plot(fgsm_eps, fgsm_rob_acc, marker='o', label='FGSM adv-trained')
plt.plot(pgd_eps, pgd_base_acc, marker='s', label='PGD baseline')
plt.plot(pgd_eps, pgd_rob_acc, marker='s', label='PGD adv-trained')
plt.xlabel('Perturbation budget ε')
plt.ylabel('Test accuracy')
plt.title('Attack impact and robustness trade-off')
plt.legend(frameon=False, ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'attack_impact.png'), dpi=220)
plt.close()

# Confusion matrices clean vs PGD baseline
from sklearn.metrics import ConfusionMatrixDisplay
cm_clean = confusion_matrix(clean_base['y_true'], clean_base['y_pred'])
cm_pgd = confusion_matrix(pgd_base['y_true'], pgd_base['y_pred'])
fig, axs = plt.subplots(1,2, figsize=(8,3.6))
ConfusionMatrixDisplay(cm_clean).plot(ax=axs[0], colorbar=False)
axs[0].set_title('Clean baseline')
ConfusionMatrixDisplay(cm_pgd).plot(ax=axs[1], colorbar=False)
axs[1].set_title('PGD attack on baseline')
plt.tight_layout()
plt.savefig(os.path.join(OUT,'confusion_compare.png'), dpi=220)
plt.close()

# Select adversarial examples
baseline.eval(); robust.eval()
selected=[]
for idx in range(len(X_test)):
    x = X_test[idx:idx+1]
    y = y_test[idx:idx+1]
    with torch.no_grad():
        pred_clean = baseline(x).argmax(1).item()
    if pred_clean != y.item():
        continue
    x_fgsm = fgsm_attack(baseline, x, y, eps=0.15)
    x_pgd = pgd_attack(baseline, x, y, eps=0.20, alpha=0.04, steps=10)
    with torch.no_grad():
        pred_fgsm = baseline(x_fgsm).argmax(1).item()
        pred_pgd = baseline(x_pgd).argmax(1).item()
        pred_fgsm_rob = robust(x_fgsm).argmax(1).item()
        pred_pgd_rob = robust(x_pgd).argmax(1).item()
    if pred_fgsm != y.item() or pred_pgd != y.item():
        selected.append((x.view(8,8).numpy(), x_fgsm.view(8,8).numpy(), x_pgd.view(8,8).numpy(), y.item(), pred_clean, pred_fgsm, pred_pgd, pred_fgsm_rob, pred_pgd_rob))
    if len(selected) == 4:
        break

fig, axs = plt.subplots(len(selected), 5, figsize=(10, 2.2*len(selected)))
if len(selected)==1:
    axs = np.array([axs])
for r,(orig,fg,pg,true,pc,pf,pp,pfr,ppr) in enumerate(selected):
    imgs=[orig, fg, pg, fg, pg]
    titles=[f'Original\ntrue={true}, pred={pc}',
            f'FGSM / baseline\npred={pf}',
            f'PGD / baseline\npred={pp}',
            f'FGSM / robust\npred={pfr}',
            f'PGD / robust\npred={ppr}']
    for c in range(5):
        axs[r,c].imshow(imgs[c], cmap='gray', vmin=0, vmax=1)
        axs[r,c].axis('off')
        axs[r,c].set_title(titles[c], fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'adversarial_examples.png'), dpi=220)
plt.close()

# Save sample metrics table CSV
import csv
with open(os.path.join(OUT,'metrics_table.csv'),'w', newline='') as f:
    w=csv.writer(f)
    w.writerow(['Model','Clean Accuracy','Macro F1','FGSM Accuracy (e=0.15)','PGD Accuracy (e=0.20)'])
    w.writerow(['Baseline MLP', clean_base['acc'], clean_base['f1'], fgsm_base['acc'], pgd_base['acc']])
    w.writerow(['Adversarially trained MLP', clean_rob['acc'], clean_rob['f1'], fgsm_rob['acc'], pgd_rob['acc']])

print(json.dumps(summary, indent=2))
