import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from models.abstract_predictor import AbstractPredictor
from models.abstract_trainer import AbstractTrainer


class DNNBinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = self.input_dim // 2
        self.dropout_rate = 0.1
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class DeepPredictor(AbstractPredictor):
    def __init__(self, path: str):
        self._load(path)

    def _load(self, path: str):
        with open(path, "rb") as f:
            checkpoint = torch.load(path)
            input_dim = checkpoint['input_dim']
            self.best_model = DNNBinaryClassifier(input_dim=input_dim)
            self.best_model.load_state_dict(checkpoint['state_dict'])
            self.best_model.eval()

    def predict(self, features: []) -> float:
        input_tensor = torch.tensor([features], dtype=torch.float32)  # shape: [1, input_dim]

        # Optional: move to GPU if needed
        # model.to("cuda")
        # input_tensor = input_tensor.to("cuda")

        # 3. Inference (no_grad for performance)
        with torch.no_grad():
            output = self.best_model(input_tensor)
        return output


class DeepTrainer(AbstractTrainer):
    def __init__(self, df):
        self.train_df = self._resampling(df)

    def save(self, path: str):
        with open(path, "wb") as f:
            torch.save({
                'input_dim': self.best_model.input_dim,
                'state_dict': self.best_model.state_dict()
            }, path)


    def train(self):
        X = self.train_df.drop(['label', 'User', 'Card'], axis=1)
        y = self.train_df['label']

        # Split 20% final test, 80% train+val
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print('data loading')
        X_train_tensor = torch.tensor(X_trainval[:100].astype(np.float32).to_numpy(), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_trainval[:100].astype(np.float32).to_numpy(), dtype=torch.float32).unsqueeze(1)  # shape [N, 1]
        X_test_tensor = torch.tensor(X_test[:100].astype(np.float32).to_numpy(), dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test[:100].astype(np.float32).to_numpy(), dtype=torch.float32).unsqueeze(1)

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

        print('data loaded')
        model = DNNBinaryClassifier(input_dim=len(X.columns))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()  # Binary Cross Entropy

        print('start training...')
        for epoch in range(10):
            model.train()
            for xb, yb in train_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                preds = model(X_test_tensor)
                pred_labels = (preds > 0.5).float()
                auc = roc_auc_score(y_test_tensor.numpy(), pred_labels.numpy())
                print(f"Epoch {epoch + 1}: Val AUC = {auc:.4f}")

        self.best_model = model
        # Final test on 20% holdout
        y_pred_proba = self.best_model(X_test_tensor)
        y_pred = (y_pred_proba > 0.5).float()
        self._evaluation(y_test_tensor.numpy(), y_pred.detach().numpy(), y_pred_proba.detach().numpy())