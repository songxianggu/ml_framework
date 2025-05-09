from models.abstract_trainer import AbstractTrainer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegressionCV
import pickle
import pandas as pd

from models.abstract_predictor import AbstractPredictor


class LogisticPredictor(AbstractPredictor):
    def __init__(self, path: str):
        self._load(path)
        pass

    def _load(self, path: str):
        with open(path, "rb") as f:
            self.best_model = pickle.load(f)

    def predict(self, features: [], feature_names: [] = []) -> float:
        return self.best_model.predict(features)


class LogisticTrainer(AbstractTrainer):
    def __init__(self, df : pd.DataFrame):
        self.train_df = self._resampling(df)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.best_model, f)

    def train(self):
        X = self.train_df.drop(['label', 'User', 'Card'], axis=1)
        y = self.train_df['label']

        # Split 20% final test, 80% train+val
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LogisticRegressionCV(
            Cs=10,  # Number of C values to try
            cv=5,  # 5-fold cross-validation
            scoring='roc_auc',  # Metric to optimize
            solver='liblinear',  # Solver for binary tasks
            penalty='l2',
            max_iter=1000,
            class_weight='balanced'
        )
        model.fit(X_trainval, y_trainval)

        self.best_model = model
        # Final test on 20% holdout
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        self._evaluation(y_test, y_pred, y_pred_proba)