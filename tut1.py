import numpy as np
from sklearn import datasets,linear_model
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.metrics import mean_squared_error
disease = datasets.load_diabetes()

# ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename']
disease_X = disease.data
disease_X_test = disease_X[-30:]
disease_X_train = disease_X[:-30]

disease_y_test = disease.target[-30:]
disease_y_train = disease.target[:-30]


model = linear_model.LinearRegression()

model.fit(disease_X_train,disease_y_train)

predict_y = model.predict(disease_X_test)

print("Mean Squared Error Is",mean_squared_error(disease_y_test,predict_y))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

# style.use('dark_background')
# plt.scatter(disease_X_test,disease_y_test)
# plt.plot(disease_X_test,predict_y)
# plt.title('Diabeties Model In Python Machince Learning')
# plt.show()

# Mean Squared Error Is 3035.0601152912695
# Weights:  [941.43097333]
# Intercept:  153.39713623331698