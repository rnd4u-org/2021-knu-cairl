def splitColumn(data, feature):
    vals = set(data[feature])
    for val in vals:
        a = []
        for i in range(data.shape[0]):
            if data[feature][i] == val:
                a.append(1)
            else:
                a.append(0)
        data[feature + "_" + str(val)] = a
    return data.drop([feature], axis=1)


def splitFloatColumn(data, feature, diapasones):
    for i in range(len(diapasones) + 1):
        a = []
        for j in range(data.shape[0]):
            t = True
            if i > 0:
                t &= (data[feature][j] >= diapasones[i - 1])
            if i < len(diapasones):
                t &= (data[feature][j] < diapasones[i])
            if t:
                a.append(1)
            else:
                a.append(0)
        data[feature + "_" + str(i)] = a
    return data.drop([feature], axis=1)


def prepare(data):
    # for val in data:
    #     if val in ['Alley', 'PoolQC', 'Fence', 'MiscFeature']:
    #       print(val)
    data = data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', "Id"], axis=1)

    for val in data:
        if data[val].dtypes == 'object':
            data[val] = data[val].fillna(data[val].mode()[0])
        else: 
            data[val] = data[val].fillna(data[val].mean())

    for val in data:
        if data[val].dtypes == 'object':
            data = splitColumn(data, val)

    return data


def fit_set(data, example):
    for val in example:
        if val not in data and val != "SalePrice":
            data[val] = [0] * data.shape[0]
    return data


def split_data(data):
    X = data.drop(['SalePrice'], axis=1)
    Y = data['SalePrice']

    from sklearn.model_selection import train_test_split

    return train_test_split(X, Y, test_size=0.2)
