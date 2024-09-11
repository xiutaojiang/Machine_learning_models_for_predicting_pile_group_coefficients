import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.optimizers import Adam
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
scaler=StandardScaler()
scaler.fit(X_train)
#Standardization
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
############################################
def create_MLP_model(neurons=6, activation='relu',learning_rate=0.001):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation=activation))
    model.add(Dense(1, activation='linear'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mse'])
    return model
model = KerasRegressor(build_fn=create_MLP_model,neurons=6, learning_rate=0.001, activation='relu', verbose=1)
neurons = [ 4, 5 ,6 ]
epochs = [ 800, 900, 1000]
activation = ['relu', 'sigmoid']
learning_rate = [0.001, 0.01]
param_grid = dict(neurons=neurons,
                  epochs=epochs,
                  learning_rate=learning_rate,
                  activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(X_train_scaled, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
############################################
pre_y_train= grid_result.predict(X_train_scaled)
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
#Predict on test data
pre_y_test= grid_result.predict(X_test_scaled)
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