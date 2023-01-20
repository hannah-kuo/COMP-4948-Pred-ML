import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# check for the sklearn version, it has to be 0.21
import sklearn
print(sklearn.__version__)
breast_cancer = datasets.load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.20)

breast_cancer = datasets.load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.20)

print(breast_cancer)

classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


def showAccuracyScores(y_test, y_pred):
    print("\nModel Evaluation")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("")
    tn = cm[0][0]
    fp = cm[0][1]
    tp = cm[1][1]
    fn = cm[1][0]
    accuracy = (tp + tn) / (tn + fp + tp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))


showAccuracyScores(y_test, y_pred)
