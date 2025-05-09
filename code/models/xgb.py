import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from models.abstract_trainer import AbstractTrainer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from models.abstract_predictor import AbstractPredictor

class XGBPredictor(AbstractPredictor):
    def __init__(self, path: str):
        self._load(path)

    def _load(self, path: str):
        self.best_model = xgb.Booster()
        self.best_model.load_model(path)

    def predict(self, features: [], feature_names: [] = []) -> float:
        features_array = np.array([features])
        dmatrix = xgb.DMatrix(features_array, feature_names=feature_names)  # Convert NumPy array to DMatrix
        prediction = self.best_model.predict(dmatrix)
        return float(prediction[0])


class XBGTrainer(AbstractTrainer):
    def __init__(self, df : pd.DataFrame):
        X = df.drop(['label', 'User', 'Card'], axis=1)
        y = df['label']

        pos = (y == 1).sum()
        neg = (y == 0).sum()

        print("pos:", pos)
        print("neg:", neg)

        self.train_df = self._resampling(df)

    def save(self, path: str):
        self.best_model.save_model(path)

    def train(self):
        X = self.train_df.drop(['label', 'User', 'Card'], axis=1)
        y = self.train_df['label']

        pos = (y == 1).sum()
        neg = (y == 0).sum()
        scale = neg / pos

        # Split 20% final test, 80% train+val
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        sample_weights = y_trainval.map(lambda x: scale if x == 1 else 1.0)
        dtrain = xgb.DMatrix(X_trainval, label=y_trainval, weight=sample_weights)

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'eta': 0.1,
            'max_depth': 4,
            'verbosity': 0
        }
        print('start training...')
        cv_result = xgb.cv(
            params,
            dtrain,
            num_boost_round=100,
            nfold=5,
            stratified=True,
            shuffle=True,
            early_stopping_rounds=10,
            metrics='auc',
            seed=42
        )

        best_round = len(cv_result)
        print(f"Best number of boosting rounds: {best_round}")


        # Final training using all data
        self.best_model = xgb.train(params, dtrain, num_boost_round=best_round)
        # Final test
        dtest = xgb.DMatrix(X_test)
        print(dtest)
        from sklearn.datasets import dump_svmlight_file
        dump_svmlight_file(X_test, y_test, '../model_data/dtest.features')
        y_pred_proba = self.best_model.predict(dtest)  # output is probability
        y_pred = (y_pred_proba > 0.5).astype(int)  # convert to binary label
        self._evaluation(y_test, y_pred, y_pred_proba)