import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def one_hot_encode(x):
    return 1 if x == 'Partial' else 0


def encode1(x):
 if (x == 4):
        return 3
 else:
        return x


if __name__ == '__main__':
    # add data in file
    train = pd.read_csv('C:/Users/Gyrocopter_UA/PycharmProjects/pythonProject/data/train.csv')
    test = pd.read_csv('C:/Users/Gyrocopter_UA/PycharmProjects/pythonProject/data/test.csv')

    # show all data
    plt.hist(train.SalePrice, color='black')
    #plt.show()

    # making some changes
    deviation = np.log(train.SalePrice)
    print("Deviation: ", deviation)
    plt.hist(deviation, color='black')
    #plt.show()

    # Now let's look at what influences the price of the house itself and analyze the data.
    influences = train.select_dtypes(include=[np.number])
    influences.dtypes
    percent = influences.corr()
    print(percent['SalePrice'].sort_values(ascending=False)[:5], '\n')
    print(percent['SalePrice'].sort_values(ascending=False)[-5:])
    quality_pivot = train.pivot_table(index='OverallQual',
                                      values='SalePrice', aggfunc=np.median)
    print(quality_pivot)
    quality_pivot.plot(kind='bar', color='black')
    plt.xlabel('Overall Quality')
    plt.ylabel('Median Sale Price')
    plt.xticks(rotation=0)
    #plt.show()

    # Thus, we got that the price is most affected by: OverallQual, GrLivArea, GarageCars,
    # GarageArea, YrSold, OverallCond, MSSubClass, EnclosedPorch, KitchenAbvGr.
    # Now let's see the rest of the factors that affect the price.

    plt.scatter(x=train['GrLivArea'], y=deviation, color="black")
    plt.ylabel('Sale Price')
    plt.xlabel('Above grade (ground) living area square feet')
    #plt.show()

    plt.scatter(x=train['GarageArea'], y=deviation, color="black")
    plt.ylabel('Sale Price')
    plt.xlabel('Garage Area')
    #plt.show()

    # Thus, we got the graphs and now it would be worth removing the points that could
    # shift our line from the "center of events".
    # Let's set the maximum value of the garage area to 1200, so that the line doesn't run away from us.
    # And makes some changes in GrLivArea


    plt.scatter(x=train['GrLivArea'], y=deviation, color="black")
    plt.ylabel('Sale Price')
    plt.xlabel('GrLivArea')
    #plt.show()

    plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice), color="black")
    plt.xlim(-1, 1600)
    plt.ylabel('Sale Price')
    plt.xlabel('Garage Area')
    #plt.show()

    train = train[train['GarageArea'] < 1200]
    train = train[train['GrLivArea'] < 4000]

    # Now let's check all NULL

    nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
    print(nulls)

    # Now let's chek data without numbers(non-numeric)

    categoricals = train.select_dtypes(exclude=[np.number])
    categoricals.describe()

    # Let's use one-hot encoding method for our non-numeric data

    train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
    test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
    condition_pivot = train.pivot_table(index='SaleCondition',
                                        values='SalePrice', aggfunc=np.median)
    condition_pivot.plot(kind='bar', color='black')
    plt.xlabel('Sale Condition')
    plt.ylabel('Median Sale Price')
    plt.xticks(rotation=0)
    #plt.show()

    train['enc_condition'] = train.SaleCondition.apply(one_hot_encode)
    test['enc_condition'] = test.SaleCondition.apply(one_hot_encode)
    train['enc_condition1'] = train.KitchenAbvGr.apply(encode1)
    test['enc_condition1'] = test.KitchenAbvGr.apply(encode1)
    condition_pivot = train.pivot_table(index='enc_condition',
                                        values='SalePrice', aggfunc=np.median)
    condition_pivot.plot(kind='bar', color='black')
    plt.xlabel('Encoded Sale Condition')
    plt.ylabel('Median Sale Price')
    plt.xticks(rotation=0)
    #plt.show()

    data = train.select_dtypes(include=[np.number]).interpolate().dropna()

    # Thus, we redid the reasons for the sale and got a simpler model, which is easier to work with both for us
    # and the computer. And also, those data that we do not use have been converted to zero, also in order to
    # make it easier for us to work.
    # Now let's start building our linear model.

    y = np.log(train.SalePrice)
    X = data.drop(['SalePrice', 'Id'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=100, test_size=.33)
    lr = linear_model.LinearRegression()
    model = lr.fit(X_train, y_train)
    predictions = model.predict(X_test)
    actual_values = y_test

    # Now let's build our line with a predicted price.

    plt.scatter(predictions, actual_values, alpha=.7,
                color='black')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Linear Regression Model')
    #plt.show()

    # Now let's change our alpha a little and get some graphics. With alpha = 1 we get better prediction.

    alpha = 1
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)
    plt.scatter(preds_ridge, actual_values, alpha=.75, color='black')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(ridge_model.score(X_test, y_test),
                                               mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay, xy=(12.1, 10.6), size='x-large')

    # Finally, we get our final answer.

    submission = pd.DataFrame()
    submission['Id'] = test.Id
    feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
    predictions = ridge_model.predict(feats)
    final_predictions = np.exp(predictions)
    print("Final predictions are: \n", final_predictions[:5])
    submission['SalePrice'] = final_predictions
    submission.head()
    submission.to_csv('submission1.csv', index=False)
    print("R^2 is: \n", model.score(X_test, y_test))


