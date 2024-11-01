import pandas as pd
import joblib
import numpy as np
import datetime

model = joblib.load('/model/profit_model.pkl')

states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 
          'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 
          'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 
          'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 
          'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 
          'VA', 'WA', 'WV', 'WI', 'WY']

start_date = datetime.date(2024, 1, 1)
date_list = [start_date + datetime.timedelta(days=i*30) for i in range(len(states) * 11)]

np.random.seed(42)  
units_sold = np.random.randint(1, 100, size=len(date_list))
costs = np.random.uniform(10, 100, size=len(date_list))

new_data = pd.DataFrame({
    'Date': date_list,
    'Units Sold': units_sold,
    'Cost': costs,
    'State': np.tile(states, len(date_list) // len(states))
})

future_predictions = model.predict(new_data[['Units Sold', 'Cost']])

predictions_df = new_data.copy()
predictions_df['Predicted Profit'] = future_predictions

predictions_df.to_csv('/model/predictions.csv', index=False)

print("Saved in 'predictions.csv'")
