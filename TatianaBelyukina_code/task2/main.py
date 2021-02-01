import pandas as pd

from pre import prepare, split_data, fit_set

houses = pd.read_csv('./train.csv')

housesTest = pd.read_csv('./test.csv')
# print(houses.head(20))


def showNanVals(data):
    for val in data:
        if data[val].isna().sum() != 0:
            print(data[val].dtypes)
            print(data[val].isna().sum(), val)


houses = prepare(houses)
housesTest = prepare(housesTest)
housesTest = fit_set(housesTest, houses)

print("HousesNan")
showNanVals(houses)
print("HousesTestNan")
showNanVals(housesTest)
print("Prepared")


x_train, x_test, y_train, y_test = split_data(houses)
test = housesTest.iloc[:, :]
print(type(x_train), type(test), type(housesTest))
print(x_train.shape, housesTest.shape)


def train(x_train, y_train, x_test, y_test):

    # import xgboost
    # classifier = xgboost.XGBRegressor()
    # classifier.fit(x_train,y_train)

    from sklearn.ensemble import RandomForestRegressor 
    classifier = RandomForestRegressor()
    classifier.fit(x_train, y_train)

    import pickle 
    filename = 'final_model.pkl'
    pickle.dump(classifier, open(filename, 'wb'))

    return classifier


model = train(x_train, y_train, x_test, y_test)

y_pred = model.predict(test)


print(y_pred)

pred = pd.DataFrame(y_pred)
sub = pd.read_csv('./sample_submission.csv')
datasets = pd.concat([sub['Id'], pred], axis=1)
datasets.columns = ['Id', 'SalePrice']
datasets.to_csv('./sample_submission.csv', index=False)
