from abc import ABC, abstractmethod
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import numpy as np

class AbstractTrainer(ABC):
    def __init__(self):
        self.best_model = None
        self.max_auc = 0
        self.test_auc = 0
        self.test_accuracy = 0
        self.test_precision = 0
        self.test_recall = 0
        self.test_f1 = 0
        self._y_pred = []
        self._y_test = []

    @abstractmethod
    def train(self, data_frame: pd.DataFrame):
        pass


    @abstractmethod
    def save(self, path: str):
        pass

    def get_accuracy(self) -> tuple[float, float]:
        return self.test_accuracy, self.ci_95

    def get_auc(self):
        return self.test_auc

    def get_precision(self):
        return self.test_precision

    def get_recall(self):
        return self.test_recall

    def get_f1(self):
        return self.test_f1


    def _resampling(self, train_df) -> pd.DataFrame:
        # Separate majority and minority class
        df_majority = train_df[train_df['label'] == 0]
        df_minority = train_df[train_df['label'] == 1]

        # Upsample minority class
        df_minority_upsampled = resample(
            df_minority,
            replace=True,  # sample with replacement
            n_samples=int(0.25 * len(df_majority)),  # match majority class size
            random_state=42
        )

        # Combine balanced data
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])

        return df_upsampled

    def _evaluation(self, y_true, y_pred, y_pred_proba):
        self.test_accuracy = accuracy_score(y_true, y_pred)
        self.test_auc = roc_auc_score(y_true, y_pred_proba)
        self.test_precision = precision_score(y_true, y_pred)
        self.test_recall = recall_score(y_true, y_pred)
        self.test_f1 = f1_score(y_true, y_pred)

        correct = (y_pred == y_true).astype(int)
        p = correct.mean()
        n = len(correct)
        variance = p * (1 - p) / n
        self.ci_95 = (p - 1.96 * np.sqrt(variance), p + 1.96 * np.sqrt(variance))

