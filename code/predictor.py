from models.logistic import LogisticPredictor
from models.xgb import XGBPredictor


if __name__ == '__main__':

    predictor = XGBPredictor('../model_data/xgb.json')
    # predicter.predict([...])

    predictor = LogisticPredictor('../model_data/logistic.pickle')
    # predicter.predict([...])