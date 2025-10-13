import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

class DirectoryPoseDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        self.label_to_idx = {}
        # Scan all subdirectories, each as a label group
        for idx, exercise in enumerate(sorted(os.listdir(root_dir))):
            exercise_dir = os.path.join(root_dir, exercise)
            if not os.path.isdir(exercise_dir):
                continue
            self.label_to_idx[exercise] = idx
            for fname in os.listdir(exercise_dir):
                if fname.endswith('.npy'):
                    self.samples.append(os.path.join(exercise_dir, fname))
                    self.labels.append(idx)
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = np.load(self.samples[idx])  # shape: [T, num_joints, 2 or 3]
        seq = seq.astype(np.float32)
        seq = seq.reshape(seq.shape[0], -1)  # [T, num_joints * (2 or 3)]
        label_idx = self.labels[idx]
        length = seq.shape[0]
        return torch.tensor(seq), torch.tensor(label_idx), length

def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True)  # [batch, max_seq_len, feat_dim]
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    return padded_sequences, labels, lengths

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, src_key_padding_mask=None):
        x = self.input_fc(x)
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x.transpose(0, 1)  # [batch, seq_len, d_model]
        return x

class TransformerLSTMClassifier(nn.Module):
    def __init__(self, input_dim, trans_d_model, trans_nhead, trans_layers, lstm_hidden, lstm_layers, num_classes, dropout=0.2):
        super().__init__()
        self.transformer = SimpleTransformerEncoder(input_dim, trans_d_model, trans_nhead, trans_layers, dropout)
        self.lstm = nn.LSTM(trans_d_model, lstm_hidden, lstm_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)
        
    def forward(self, x, lengths):
        batch_size = x.size(0)
        max_len = x.size(1)
        mask = torch.arange(max_len)[None, :].to(lengths.device) >= lengths[:, None]
        x = self.transformer(x, src_key_padding_mask=mask)
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        idx = (lengths - 1).view(-1, 1).expand(batch_size, out.size(2)).unsqueeze(1)
        last_outputs = out.gather(1, idx).squeeze(1)
        logits = self.fc(last_outputs)
        return logits

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for seqs, labels, lengths in dataloader:
            seqs, labels, lengths = seqs.to(device), labels.to(device), lengths.to(device)
            outputs = model(seqs, lengths)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(list(preds))
            all_labels.extend(list(labels.cpu().numpy()))
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print("Classification Report:\n", classification_report(all_labels, all_preds))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    return acc, f1

# Hyperparameters
DATA_DIR = 'path/to/data'  # Directory with exercise subfolders
BATCH_SIZE = 8
TRANS_D_MODEL = 128
TRANS_NHEAD = 4
TRANS_LAYERS = 2
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
NUM_EPOCHS = 20
VAL_SPLIT = 0.1
RANDOM_SEED = 42

# Prepare data
dataset = DirectoryPoseDataset(DATA_DIR)
input_dim = dataset[0][0].shape[1]
num_classes = len(dataset.label_to_idx)

# Split train/val
np.random.seed(RANDOM_SEED)
indices = np.random.permutation(len(dataset))
split = int(len(dataset) * (1 - VAL_SPLIT))
train_idx, val_idx = indices[:split], indices[split:]
train_set = torch.utils.data.Subset(dataset, train_idx)
val_set = torch.utils.data.Subset(dataset, val_idx)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerLSTMClassifier(
    input_dim, TRANS_D_MODEL, TRANS_NHEAD, TRANS_LAYERS,
    LSTM_HIDDEN, LSTM_LAYERS, num_classes
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for seqs, labels, lengths in train_loader:
        seqs, labels, lengths = seqs.to(device), labels.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(seqs, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {total_loss/len(train_loader):.4f}")
    val_acc, val_f1 = evaluate(model, val_loader, device)
    print(f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

# Final evaluation
print("Final Evaluation on Validation Set:")
val_acc, val_f1 = evaluate(model, val_loader, device)
torch.save(model.state_dict(), "rehab246_transformer_lstm_classifier.pth")