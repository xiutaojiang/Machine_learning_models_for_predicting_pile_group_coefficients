import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import matplotlib.pyplot as plt
############################################
# load data
df = pd.read_excel('Group_coefficient_total.xlsx',sheet_name='Sheet1')
feature_names = ['Iribarren_number', 'Wave_direction', 'Spacing', 'Group_coefficient']
print(df.head())
df.columns = feature_names
print(df.head())
print(df.describe())
############################################
#Split into features and target
X = df.drop('Group_coefficient', axis = 1)
y = df['Group_coefficient']
X = X.values
y = y.values
#Split into train\test\validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
scaler=StandardScaler()
scaler.fit(X_train)
#Standardization
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# XGBRegressor model
from xgboost import XGBRegressor
model =  XGBRegressor()
# grid_search
param_grid = {
     'learning_rate': [0.1, 0.2, 0.3],
     'max_depth': [3, 4, 5],
     'n_estimators': [40, 60, 80, 100],
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error',cv=5)
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
# # Use the best model to make predictions on the training dataset
pre_y_train = best_model.predict(X_train_scaled)
y_train_MAE=mean_absolute_error(y_train,pre_y_train)
y_train_MSE=mean_squared_error(y_train,pre_y_train)
y_train_R2=r2_score(y_train,pre_y_train)
print("y_train_MAE:", y_train_MAE)
print("y_train_MSE:", y_train_MSE)
print("y_train_R2:", y_train_R2)
plt.scatter(y_train, pre_y_train)
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
plt.plot([ymin, ymax], [ymin, ymax], "r--", lw=1, alpha=0.6)
plt.xlabel("True y_train")
plt.ylabel("Predicted y_train")
plt.title('Predicted y_train vs. True y_train')
plt.text(min(y_train), max(pre_y_train), f"MSE: {y_train_MSE:.2f}", ha='left', va='bottom')
plt.text(min(y_train), max(pre_y_train) - 0.1 * (max(pre_y_train) - min(y_train)), f"MAE: {y_train_MAE:.2f}", ha='left', va='center')
plt.text(min(y_train), max(pre_y_train) - 0.2 * (max(pre_y_train) - min(y_train)), f"R2: {y_train_R2:.2f}", ha='left', va='top')
plt.show(block=True)
# # Use the best model to make predictions on the training dataset
pre_y_test = best_model.predict(X_test_scaled)
y_test_MAE=mean_absolute_error(y_test,pre_y_test)
y_test_MSE=mean_squared_error(y_test,pre_y_test)
y_test_R2=r2_score(y_test,pre_y_test)
print("y_test_MAE:", y_test_MAE)
print("y_test_MSE:", y_test_MSE)
print("y_test_R2:", y_test_R2)
plt.scatter(y_test, pre_y_test)
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
plt.plot([ymin, ymax], [ymin, ymax], "r--", lw=1, alpha=0.6)
plt.xlabel("True y_test")
plt.ylabel("Predicted y_test")
plt.title('Predicted y_test vs. True y_test')
plt.text(min(y_test), max(pre_y_test), f"MSE: {y_test_MSE:.2f}", ha='left', va='bottom')
plt.text(min(y_test), max(pre_y_test) - 0.1 * (max(pre_y_test) - min(y_test)), f"MAE: {y_test_MAE:.2f}", ha='left', va='center')
plt.text(min(y_test), max(pre_y_test) - 0.2 * (max(pre_y_test) - min(y_test)), f"R2: {y_test_R2:.2f}", ha='left', va='top')
plt.show(block=True)