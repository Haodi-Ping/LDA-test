from __future__ import print_function
import time
init_st = time.time() * 1000
import bentoml
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import custom model class
from lda import LDA
init_ed = time.time() * 1000
def main(request):
    fun_st = time.time() * 1000
    # Load the dataset
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Three -> two classes
    X = X[y != 2]
    y = y[y != 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Fit and predict using LDA
    lda = LDA()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

    # Save model with BentoML
    saved_model = bentoml.picklable_model.save_model(
        'iris_clf_lda',
        lda,
        signatures={"predict": {"batchable": True}},
    )
    print(f"Model saved: {saved_model}")

    fun_ed = time.time() * 1000
    
    return ',InitStart:{},'.format(init_st) + 'InitEnd:{},'.format(init_ed) + 'functionStart:{},'.format(fun_st) + 'functionEnd:{},'.format(fun_ed)

# main("")
# if __name__ == "__main__":
#     main()