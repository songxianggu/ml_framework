from models.logistic import LogisticPredictor
from models.xgb import XGBPredictor
from sklearn.datasets import load_svmlight_file

feature_names = ["MCC","Per Capita Income - Zipcode","Yearly Income - Person","Total Debt","FICO Score","Num Credit Cards","Credit Limit","Year PIN last Changed","Errors?_Bad CVV","Errors?_Bad CVV,Insufficient Balance","Errors?_Bad CVV,Technical Glitch","Errors?_Bad Card Number","Errors?_Bad Card Number,Insufficient Balance","Errors?_Bad Expiration","Errors?_Bad Expiration,Bad CVV","Errors?_Bad Expiration,Insufficient Balance","Errors?_Bad Expiration,Technical Glitch","Errors?_Bad PIN","Errors?_Bad PIN,Insufficient Balance","Errors?_Bad PIN,Technical Glitch","Errors?_Bad Zipcode","Errors?_Insufficient Balance","Acct Open Date_1991","Acct Open Date_1992","Acct Open Date_1994","Acct Open Date_1995","Acct Open Date_1996","Acct Open Date_1998","Acct Open Date_1999","Acct Open Date_2000","Acct Open Date_2002","Acct Open Date_2005","Acct Open Date_2006","Acct Open Date_2007","Acct Open Date_2008","Acct Open Date_2009","Acct Open Date_2010","Acct Open Date_2013","Acct Open Date_2017","Acct Open Date_2019","Acct Open Date_2020"]

if __name__ == '__main__':

    X_loaded, y_loaded = load_svmlight_file('../model_data/dtest.features')
    X_dense = X_loaded.toarray()
    predictor = XGBPredictor('../model_data/xgb.json')
    for i in range(len(X_dense)):
        x_row = X_dense[i]
        pred = predictor.predict(x_row, feature_names=feature_names)
        print(f"Input {i} → Prediction: {pred}")

    # predictor = LogisticPredictor('../model_data/logistic.pickle')
    # for i in range(len(X_dense)):
    #     x_row = X_dense[i].reshape(1, -1)  # shape [1, n_features]
    #     pred = predictor.predict(x_row)
    #     print(f"Input {i} → Prediction: {pred[0]}")