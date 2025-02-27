import pandas as pd   # Great for tables (google spreadsheets, microsoft excel, csv).
import os # Good for navigating your computer's files
import warnings
warnings.filterwarnings("ignore")

# read our data in using 'pd.read_csv('file')'
data_path  = 'linear_Regression/car_dekho.csv'
car_data = pd.read_csv(data_path)

# Exploring the Data
car_data.head()

# Visualizing the Data
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x='Age', y='Selling_Price', data=car_data)

# Visualizing Categorical Data
sns.catplot(x='Fuel_Type', y='Selling_Price', data=car_data, kind='swarm', s=2)

# Groupby Exercise
car_data.groupby(['Fuel_Type']).count()

# Scatter plot for Kms_Driven vs. Selling_Price
sns.scatterplot(x='Kms_Driven', y='Selling_Price', data=car_data)

# Categorical plots for Seller_Type and Transmission
sns.catplot(x='Seller_Type', y='Selling_Price', data=car_data, kind='swarm', s=2)
sns.catplot(x='Transmission', y='Selling_Price', data=car_data, kind='swarm', s=2)

# Linear Regression
from sklearn.linear_model import LinearRegression
import numpy as np

X = car_data[['Age']]
y = car_data[['Selling_Price']]

linear = LinearRegression()
linear.fit(X, y)

y_pred = linear.predict(X)
plt.plot(X, y_pred, color='red')
plt.scatter(X, y)
plt.xlabel('Age')
plt.ylabel('Selling_Price (lakhs)')
plt.show()

print('Our m in lakhs/year: ', linear.coef_)
print('Our intercept b: ', linear.intercept_)

# Optional: Single Linear Regression with different inputs
car_data['TransmissionNumber'] = car_data['Transmission'].replace({'Manual': 1, 'Automatic': 0})
X_column = 'Age'
X = car_data[[X_column]]
y = car_data[['Selling_Price']]

linear = LinearRegression()
linear.fit(X, y)

y_pred = linear.predict(X)
plt.plot(X, y_pred, color='red')
plt.scatter(X, y)
plt.xlabel(X_column)
plt.ylabel('Selling_Price (lakhs)')
plt.show()

# Multiple Linear Regression
X = car_data[['Age', 'TransmissionNumber', 'Kms_Driven']]
multiple = LinearRegression(fit_intercept=True)
multiple.fit(X, y)

print('Our single linear model had an R^2 of: %0.3f' % linear.score(car_data[[X_column]], y))
print('Our multiple linear model had an R^2 of: %0.3f' % multiple.score(X, y))

# Challenge Section: Finding The Best Deal
prediction = multiple.predict(X)
plt.plot([-5, 15], [-5, 15])
plt.title("Predicted vs. Real Prices")
plt.xlabel("Real price")
plt.ylabel("Predicted price")
plt.scatter(y, prediction)
plt.show()

car_data['Prediction'] = prediction
car_data['deal_score'] = car_data['Selling_Price'] - car_data['Prediction']
best_deals = car_data.sort_values(by='deal_score', ascending=False).head(10)
most_overpriced = car_data.sort_values(by='deal_score').head(10)

print(best_deals)
print(most_overpriced)

plt.plot([-5, 15], [-5, 15])
plt.title("Predicted vs. Real Prices")
plt.xlabel("Real price")
plt.ylabel("Predicted price")
plt.scatter(y, prediction, label='Other cars', alpha=0.5)
plt.scatter(best_deals['Selling_Price'], best_deals['Prediction'], label='Best deals', color='green')
plt.scatter(most_overpriced['Selling_Price'], most_overpriced['Prediction'], label='Most overpriced', color='red')
plt.legend()
plt.show()