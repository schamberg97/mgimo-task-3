# Load libraries
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pandas as pd #Import pandas library
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

results = []
names = []

# Загрузка данных
def read_data():
    DF = pd.read_csv('bank-full.csv', delimiter=";", true_values=["success","yes"], false_values=["failure","no"])
    DF = feature_to_dummy(DF, "education", True)
    DF = feature_to_dummy(DF, "marital", True)
    DF = feature_to_dummy(DF, "job", True)
    DF = feature_to_dummy(DF, "contact", True)
    return DF

#Выбор фиктивных переменных
def feature_to_dummy(DF, column, drop=False):
    tmp = pd.get_dummies(DF[column], prefix=column, prefix_sep='_')
    DF = pd.concat([DF, tmp], axis=1)
    DF.reindex()
    if drop:
        del DF[column]
    return DF
#Реализация обучения
def implement_machine_learning(DF):
    features = ["age", 'job_student', 'job_unemployed',
            "marital_divorced", "marital_married", "marital_single",
                "education_secondary", "education_tertiary", "default","balance","housing", "loan", "campaign"]
    validation_size = 0.20
    seed = 7
    scoring = 'accuracy'
    Y = DF['y'] #Выбор y из столбца Result
    X = DF[features] #Выбор x (остальные)
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Решение с помощью метода ближайших соседей
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

    # Тестирование на различных метриках
    models = []
    LDA = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False, tol=0.001)
    models.append(('LR', LogisticRegression(solver='liblinear')))
    models.append(('LDA', LDA))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    # Применение каждой модели

    for name, model in models:
        kfold = KFold(n_splits=10)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    LDA.fit(X_train, Y_train)
    return LDA


def compare_algorithms():
    # Сравнение алгоритмов
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


if __name__ == "__main__":
    DF = read_data()
    implement_machine_learning(DF)
    compare_algorithms()