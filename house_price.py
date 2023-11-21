import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
import pickle
warnings.filterwarnings("ignore")

df = pd.read_csv('Housing.csv')
# print(df)

arr = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
for i in arr:
  df[f'{i}'].replace(['yes', 'no'], [1, 0], inplace=True)
df.furnishingstatus.replace(
    ['unfurnished', 'semi-furnished', 'furnished'], [0, 1, 2], inplace=True)


# Spliting of dataset
x = df.drop('price', axis=1)  # Features
y = df.price                # Target variable

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

hp_pred = LinearRegression()


hp_pred.fit(x_train, y_train)

pred_y = hp_pred.predict(x_test)

# a = [4000, 4, 2, 5, 1, 1, 0, 0, 1, 3, 1, 2]
# input=[x for x in a]
# final=[np.array(input)]

# b = hp_pred.predict(final)

# print("\nAcurracy of the model is",r2_score(y_test,pred_y)*100,'%\n')

# print(f'For such input {a} --> Predicted House Price : {b} /- Rs.')


pickle.dump(hp_pred, open('model.pkl', 'wb'))