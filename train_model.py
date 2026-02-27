import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("health_data_labeled.csv")

X = df[['heart_rate','spo2','temperature','activity','hrv']]
y = df['risk']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train,y_train)

accuracy = model.score(X_test,y_test)
print("Model Accuracy:", accuracy)

pickle.dump(model, open("health_model.pkl","wb"))