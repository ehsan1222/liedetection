from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


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


def support_vector_machine_classifier(x_train, y_train, x_test):
    s_v_m = svm()
    s_v_m.fit(x_train, y_train)
    y_predict = s_v_m.predict(x_test)
    return y_predict


def k_nearest_neighbor_classifier(x_train, y_train, x_test):
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    y_predict = knn.predict(x_test)
    return y_predict
