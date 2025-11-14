import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

data = pd.read_csv('data/scoring.csv')
x = data.drop(['default'], axis = 1).values
y = data['default'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression(class_weight='balanced')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(f'precision {precision_score(y_test, y_pred) * 100:.0f}%, recall {recall_score(y_test, y_pred)* 100:.0f}%')
print(f'falls {y_pred.mean() * 100:.0f}%')

joblib.dump(model, 'model.pkl')