
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as score
import pandas as pd

# Funcion para calcular la especificidad
def specificity_score(y_test, preds):
    conf_matrix = score.confusion_matrix(y_test, preds)
    true_negatives = conf_matrix[0, 0]
    false_positives = conf_matrix[0, 1]
    return true_negatives / (true_negatives + false_positives)

print("############################### Datos: Calidad de Vinos #############################################")
dir_csv = '.\\archive.ics.uci.edu_ml_machine-learning-databases_wine-quality_winequality-white.csv'

data = pd.read_csv(dir_csv)
print(data.head())
#print(data.info())

# Dividir los datos en X y Y
X = data.iloc[:, :-1]  # Tomamos la ultima columna como nuestro target
y = data.iloc[:, -1]

# Dividir en sets de prueba y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar los parametros para knn y svm
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar calificadores
logistic_regression = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=10)
svm = SVC(kernel='linear')
naive_bayes = GaussianNB()

logistic_regression.fit(X_train, y_train)
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)
naive_bayes.fit(X_train, y_train)

# Predicciones
logistic_regression_pred = logistic_regression.predict(X_test)
knn_pred = knn.predict(X_test)
svm_pred = svm.predict(X_test)
naive_bayes_pred = naive_bayes.predict(X_test)

# Imprimir accuracy
print("Logistic Regression Accuracy:", score.accuracy_score(y_test, logistic_regression_pred))
print("K-Nearest Neighbors Accuracy:", score.accuracy_score(y_test, knn_pred))
print("Support Vector Machines Accuracy:", score.accuracy_score(y_test, svm_pred ))
print("Naive Bayes Accuracy:", score.accuracy_score(y_test, naive_bayes_pred))

print("\nLogistic Regression F1:", score.f1_score(y_test, logistic_regression_pred, average='micro'))
print("K-Nearest Neighbors F1:", score.f1_score(y_test, knn_pred, average='micro'))
print("Support Vector Machines F1:", score.f1_score(y_test, svm_pred , average='micro'))
print("Naive Bayes F1:", score.f1_score(y_test, naive_bayes_pred, average='micro'))

print("\nLogistic Regression Precision:", score.precision_score(y_test, logistic_regression_pred, average='micro'))
print("K-Nearest Neighbors Precision:", score.precision_score(y_test, knn_pred, average='micro'))
print("Support Vector Machines Precision:", score.precision_score(y_test, svm_pred , average='micro'))
print("Naive Bayes Precision:", score.precision_score(y_test, naive_bayes_pred, average='micro'))

print("\nLogistic Regression Sensibilidad:", score.recall_score(y_test, logistic_regression_pred, average='micro'))
print("K-Nearest Neighbors Sensibilidad:", score.recall_score(y_test, knn_pred, average='micro'))
print("Support Vector Machines Sensibilidad:", score.recall_score(y_test, svm_pred , average='micro'))
print("Naive Bayes Sensibilidad:", score.recall_score(y_test, naive_bayes_pred, average='micro'))

print("\nLogistic Regression Especificidad:", score.recall_score(y_test, logistic_regression_pred, average='micro'))
print("K-Nearest Neighbors Especificidad:", score.recall_score(y_test, knn_pred, average='micro'))
print("Support Vector Machines Especificidad:", score.recall_score(y_test, svm_pred, average='micro'))
print("Naive Bayes Especificidad:", score.recall_score(y_test, naive_bayes_pred, average='micro'))







print("############################### Datos: Seguros Vehicular #############################################")

data = pd.read_csv('.\\automakers.csv', dtype=float)
print(data.head())
print("No se puede, no son datos clasificables")

print("############################### Datos: Diabetes #############################################")

data = pd.read_csv('.\\raw.githubusercontent.com_jbrownlee_Datasets_master_pima-indians-diabetes.csv', dtype=float, header=None)
print(data.head())
#print(data.info())

# Dividir los datos en X y Y
X = data.iloc[:, :-1]  # Tomamos la ultima columna como nuestro target
y = data.iloc[:, -1]

# Dividir en sets de prueba y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Estandarizar los parametros para knn y svm
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar calificadores
logistic_regression = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=10)
svm = SVC(kernel='linear')
naive_bayes = GaussianNB()

logistic_regression.fit(X_train, y_train)
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)
naive_bayes.fit(X_train, y_train)

# Predicciones
logistic_regression_pred = logistic_regression.predict(X_test)
knn_pred = knn.predict(X_test)
svm_pred = svm.predict(X_test)
naive_bayes_pred = naive_bayes.predict(X_test)

print("\n")

# Imprimir accuracy
print("Logistic Regression Accuracy:", score.accuracy_score(y_test, logistic_regression_pred))
print("K-Nearest Neighbors Accuracy:", score.accuracy_score(y_test, knn_pred))
print("Support Vector Machines Accuracy:", score.accuracy_score(y_test, svm_pred ))
print("Naive Bayes Accuracy:", score.accuracy_score(y_test, naive_bayes_pred))

print("\nLogistic Regression F1:", score.f1_score(y_test, logistic_regression_pred, average='micro'))
print("K-Nearest Neighbors F1:", score.f1_score(y_test, knn_pred, average='micro'))
print("Support Vector Machines F1:", score.f1_score(y_test, svm_pred , average='micro'))
print("Naive Bayes F1:", score.f1_score(y_test, naive_bayes_pred, average='micro'))

print("\nLogistic Regression Precision:", score.precision_score(y_test, logistic_regression_pred, average='micro'))
print("K-Nearest Neighbors Precision:", score.precision_score(y_test, knn_pred, average='micro'))
print("Support Vector Machines Precision:", score.precision_score(y_test, svm_pred , average='micro'))
print("Naive Bayes Precision:", score.precision_score(y_test, naive_bayes_pred, average='micro'))

print("\nLogistic Regression Sensibilidad:", score.recall_score(y_test, logistic_regression_pred, average='micro'))
print("K-Nearest Neighbors Sensibilidad:", score.recall_score(y_test, knn_pred, average='micro'))
print("Support Vector Machines Sensibilidad:", score.recall_score(y_test, svm_pred , average='micro'))
print("Naive Bayes Sensibilidad:", score.recall_score(y_test, naive_bayes_pred, average='micro'))

print("\nLogistic Regression Especificidad:", score.recall_score(y_test, logistic_regression_pred, average='micro'))
print("K-Nearest Neighbors Especificidad:", score.recall_score(y_test, knn_pred, average='micro'))
print("Support Vector Machines Especificidad:", score.recall_score(y_test, svm_pred, average='micro'))
print("Naive Bayes Especificidad:", score.recall_score(y_test, naive_bayes_pred, average='micro'))

