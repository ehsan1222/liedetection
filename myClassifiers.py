from sklearn.naive_bayes import GaussianNB


def naiveBayesClassifier(x_train, y_train, x_test):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_predict = gnb.predict(x_test)
    return y_predict
