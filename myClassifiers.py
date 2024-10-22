from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


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
    s_v_m = SVC(gamma='auto')
    s_v_m.fit(x_train, y_train)
    y_predict = s_v_m.predict(x_test)
    return y_predict


def k_nearest_neighbor_classifier(x_train, y_train, x_test):
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(x_train, y_train)
    y_predict = knn.predict(x_test)
    return y_predict


def multi_layer_perceptron_classifier(x_train, y_train, x_test):
    mlp = MLPClassifier(hidden_layer_sizes=(200, 150), activation="logistic", max_iter=1000)
    mlp.fit(x_train, y_train)
    y_predict = mlp.predict(x_test)
    return y_predict
