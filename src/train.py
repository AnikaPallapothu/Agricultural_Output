# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
from data_preparation import prepare_data
import warnings
import pickle
warnings.filterwarnings('ignore')

# Load the Titanic dataset
df_train = pd.read_csv("../data/agricultural_yield_train.csv")
df_test = pd.read_csv("../data/agricultural_yield_test.csv")


xtrain, ytrain = prepare_data(df_train)
xtest, ytest = prepare_data(df_test)

# Create a Random Forest Classifier
rf_classifier = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(xtrain, ytrain)

# Make predictions on the test set
ypred = rf_classifier.predict(xtest)

# Evaluate the model
r2score = r2_score(ytest,ypred)
print(r2score)

with open('../model/yield_model_rf.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)