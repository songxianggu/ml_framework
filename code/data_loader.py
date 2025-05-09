import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import utils

user_file = '../data/sd254_users.csv'
card_file = '../data/sd254_cards.csv'
credit_files = '../data/User0_credit_card_transactions.csv'  #'../data/User0_credit_card_transactions.csv' #'./data/credit_card_transactions-ibm_v2.csv'

class DataLoader():
    # for even larger file, we need to stream this 'credit_files'
    def __init__(self, user_file : str, card_file : str, credit_files : str):
        users_df = pd.read_csv(user_file)
        users_df.head()
        users_df = users_df.reset_index()
        users_df = users_df.rename(columns={'index': 'User'})
        #['Person', 'Current Age', 'Retirement Age', 'Birth Year', 'Birth Month', 'Gender', 'Address', 'Apartment', 'City', 'State', 'Zipcode', 'Latitude', 'Longitude', 'Per Capita Income - Zipcode', 'Yearly Income - Person', 'Total Debt', 'FICO Score', 'Num Credit Cards']

        cards_df = pd.read_csv(card_file)
        cards_df.head()
        cards_df = cards_df.rename(columns={'CARD INDEX': 'Card'})
        #['User', 'CARD INDEX', 'Card Brand', 'Card Type', 'Card Number', 'Expires', 'CVV', 'Has Chip', 'Cards Issued', 'Credit Limit', 'Acct Open Date', 'Year PIN last Changed', 'Card on Dark Web']

        # When select features, make sure there is not discrimination. So let's skip age, gender, location and occupation
        # Also skip all the PII data.
        # All PII data could be used to create an identical user id using MD5 or other hash function for user alignment in federated learning.
        users_cards_df = pd.merge(cards_df, users_df, on='User', how='inner')[['User', 'Card', 'Per Capita Income - Zipcode', 'Yearly Income - Person', 'Total Debt', 'FICO Score', 'Num Credit Cards', 'Credit Limit', 'Acct Open Date', 'Year PIN last Changed', 'Card on Dark Web']]
        credits_df = pd.read_csv(credit_files)
        credits_df.head()
        print(users_cards_df.columns.tolist())
        #['User', 'Card', 'Year', 'Month', 'Day', 'Time', 'Amount', 'Use Chip', 'Merchant Name', 'Merchant City', 'Merchant State', 'Zip', 'MCC', 'Errors?', 'Is Fraud?']

        # We can calculate more features here like '500 miles away since yesterday?', '5 Errors in one day?'. We can add later.
        self.credits_users_cards_df = pd.merge(credits_df, users_cards_df, on=['User', 'Card'], how='inner')[['User', 'Card', 'MCC', 'Errors?', 'Per Capita Income - Zipcode', 'Yearly Income - Person', 'Total Debt', 'FICO Score', 'Num Credit Cards', 'Credit Limit', 'Acct Open Date', 'Year PIN last Changed', 'Card on Dark Web', 'Is Fraud?']]
        print(self.credits_users_cards_df.columns.tolist())

        self._clean_features()

        # set up label.
        self.credits_users_cards_df['label'] = self.credits_users_cards_df['Is Fraud?'].map({'Yes': 1, 'No': 0})
        self.credits_users_cards_df.drop(columns=['Is Fraud?'], inplace=True)
        self.credits_users_cards_df = self.credits_users_cards_df.dropna(subset=['label'])

    # feature cleaning
    # now we need to clean the features one by one
    # since I plan to use XGBoost
    # 1. category features need to be converted to one-hot vector.
    # 2. fill holes in the features.
    # I am not sure I have time to correct all 12 features, let me fix as many as possible.
    def _reshape_category_features(self, feature_name : str):
        one_hot = pd.get_dummies(self.credits_users_cards_df[feature_name], prefix=feature_name)
        self.credits_users_cards_df = pd.concat([self.credits_users_cards_df.drop(columns=[feature_name]), one_hot], axis=1)

    # only applicable to number features
    # To simplify the issue, I only fill in mean value here.
    # But actually, we have some strategies like mean/ row similarity/ drop feature
    def _fill_nan_features(self, feature_name : str):
        self.credits_users_cards_df[feature_name] = self.credits_users_cards_df[feature_name].fillna(self.credits_users_cards_df[feature_name].mean())

    def _clean_features(self):
        # feature: Error
        self._reshape_category_features('Errors?')
        # feature: Acct Open Date
        self.credits_users_cards_df['Acct Open Date'] = self.credits_users_cards_df['Acct Open Date'].str[-4:]
        self._reshape_category_features('Acct Open Date')
        # feature: MCC
        self._fill_nan_features('MCC')
        self.credits_users_cards_df['Yearly Income - Person'] = self.credits_users_cards_df['Yearly Income - Person'].str[1:].astype(int)
        self._fill_nan_features('Yearly Income - Person')
        self.credits_users_cards_df['Total Debt'] = self.credits_users_cards_df['Total Debt'].str[1:].astype(int)
        self._fill_nan_features('Total Debt')
        self.credits_users_cards_df['Per Capita Income - Zipcode'] = self.credits_users_cards_df['Per Capita Income - Zipcode'].str[1:].astype(int)
        self._fill_nan_features('Per Capita Income - Zipcode')
        self.credits_users_cards_df['Credit Limit'] = self.credits_users_cards_df['Credit Limit'].str[1:].astype(int)
        self._fill_nan_features('Credit Limit')
        self._fill_nan_features('Num Credit Cards')
        self._fill_nan_features('Year PIN last Changed')
        self._fill_nan_features('FICO Score')
        self.credits_users_cards_df['Card on Dark Web'] = (
            self.credits_users_cards_df['Card on Dark Web']
            .fillna('No')  # fill nulls first
            .map({'Yes': 1, 'No': 0})  # then map to 1/0
        )

    def get_original_training_data(self) -> pd.DataFrame:
        return self.credits_users_cards_df

    def get_refined_training_data(self, chi_square_threshold : float) -> pd.DataFrame:
        feature_names = self.credits_users_cards_df.columns.tolist()[2:-1]
        refined_credits_users_cards_df = self.credits_users_cards_df.copy()

        for feature_name in feature_names:
            # calculate chi-square score
            chi_squre = utils.calculate_chi_square(refined_credits_users_cards_df[feature_name], refined_credits_users_cards_df['label'])
            print(feature_name, chi_squre)
            if chi_squre < chi_square_threshold:
                refined_credits_users_cards_df.drop(columns=[feature_name], inplace=True)
        return refined_credits_users_cards_df


if __name__ == '__main__':
    data_loader = DataLoader(user_file, card_file, credit_files)
    df = data_loader.get_refined_training_data(1e-7)
    print(df.columns.tolist())