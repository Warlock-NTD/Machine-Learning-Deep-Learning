from sklearn import datasets, linear_model
import LinearRegression as LR
# fit the model by linear regression
regress = linear_model.LinearRegression()

regress.fit(LR.X, LR.y)

print("scikit-learn’s solution: w_1 = ", regress.coef_[0], "w_0 = ",\
regress.intercept_)
print("our solution : w_1 = ", LR.res[1], "w_0 = ", LR.res[0])