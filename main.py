import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import  preprocessing

from knn import knn


if __name__ == '__main__':
    data = pd.read_csv('car.data')

    # converting to numeric data:
    le = preprocessing.LabelEncoder()
    buying = le.fit_transform(list(data['buying']))
    maint = le.fit_transform(list(data['maint']))
    door = le.fit_transform(list(data['door']))
    persons = le.fit_transform(list(data['persons']))
    lug_boot = le.fit_transform(list(data['lug_boot']))
    safety = le.fit_transform(list(data['safety']))
    cls = le.fit_transform(list(data['class']))

    predict = 'class'

    x = list(zip(buying, maint, door, persons, lug_boot, safety))
    y = list(cls)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
    k = 5

    # sklearn:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    sklearn_acc = model.score(x_test, y_test)
    print("Sklearn accuracy: ", sklearn_acc)

    # my knn:
    knn_predictions = knn(x_train, y_train, x_test, k)
    my_knn_acc = sum(1.0 for p, y in zip(knn_predictions, y_test) if p == y) / float(len(y_test))
    print("My KNN accuracy: ", my_knn_acc)