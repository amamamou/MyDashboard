import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

file_path = '/model/Test Start recrutement oz (1) (1).xlsx'  
dataset = pd.read_excel(file_path)

X = dataset[['Units Sold', 'Cost']]  
y = dataset['Total Amount']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, '/model/profit_model.pkl')  

print("Model trained and saved as 'profit_model.pkl'")
