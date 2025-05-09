from data_loader import DataLoader
from models.logistic import LogisticTrainer
from models.xgb import XBGTrainer
from models.deep import DeepTrainer
import pandas as pd

user_file = '../data/sd254_users.csv'
card_file = '../data/sd254_cards.csv'
credit_files = '../data/credit_card_transactions-ibm_v2.csv' #'../data/User0_credit_card_transactions.csv' #'./data/credit_card_transactions-ibm_v2.csv'


if __name__ == '__main__':
    # data_loader = DataLoader(user_file, card_file, credit_files)
    # df = data_loader.get_refined_training_data(1e-7)
    # print('start saving training data.')
    # df.to_csv('../data/refined_training_data.csv', index=False)
    # print('finished saving training data.')

    df = pd.read_csv('../data/refined_training_data.csv')

    trainer = XBGTrainer(df)
    trainer.train()
    # trainer.save('../model_data/xgb.json')
    print(trainer.test_auc)
    print(trainer.test_accuracy)

    # trainer = DeepTrainer(df)
    # trainer.train()
    # trainer.save('../model_data/deep.pth')
    # print(trainer.test_auc)
    # print(trainer.get_accuracy())

    trainer = LogisticTrainer(df)
    trainer.train()
    trainer.save('../model_data/logistic.pickle')
    print(trainer.test_auc)
    print(trainer.test_accuracy)

