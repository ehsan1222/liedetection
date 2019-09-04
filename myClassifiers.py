from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def naive_bayes_classifier(x_train, y_train, x_test):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_predict = gnb.predict(x_test)
    return y_predict


def linear_discriminant_analysis_classifier(x_train, y_train, x_test):
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    y_predict = lda.predict(x_test)
    return y_predict
