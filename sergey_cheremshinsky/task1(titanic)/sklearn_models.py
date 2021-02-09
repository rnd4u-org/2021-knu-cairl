def sklearn_models(x_train, y_train, x_test, y_test):
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion="entropy")
    tree.fit(x_train, y_train)

    from sklearn.linear_model import LogisticRegression
    logReg = LogisticRegression(max_iter=1000000)
    logReg.fit(x_train, y_train)

    from sklearn.neighbors import KNeighborsClassifier
    neighbors = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    neighbors.fit(x_train, y_train)

    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10, criterion="entropy")
    forest.fit(x_train, y_train)

    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(x_train, y_train)

    from sklearn.svm import SVC
    svc = SVC(kernel="linear")
    svc.fit(x_train, y_train)

    svcRBF = SVC(kernel="rbf")
    svcRBF.fit(x_train, y_train)

    return [tree, forest, logReg, gauss, svc]


def models_predict(models, x):
    rez = sum(model.predict(x) for model in models) / len(models)
    for i in range(len(rez)):
        if rez[i] < 0.5:
            rez[i] = 0
        else:
            rez[i] = 1
    return rez.astype(int)


def boosted_test(models, x_test, y_test):
    for model in models:
        print(model.score(x_test, y_test))
    rez = models_predict(models, x_test)
    k = t = 0

    for i in range(len(y_test)):
        k += 1
        if rez[i] == y_test[i]:
            t += 1
    print(k, t, t / k)
    return t / k
