import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel
###################################################
############     Removing Features with low var ###
def remove_with_var_thresh(X,thresh):
    sel = VarianceThreshold(threshold=thresh)
    return sel.fit_transform(X)

def  Univariate_feature_selection(X,y):
    return SelectKBest(chi2, k=2).fit_transform(X, y)
def recursive_feature_elim(X,y):

    y = np.round(y*10)
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
                scoring='accuracy')
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


def L1_based_selection(X,y):
    
    y = np.round(y*10)
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)
    return model.transform(X)
def tree_selection(X,y):

    y = np.round(y*10)
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X, y)
    # clf.feature_importances_  
    model = SelectFromModel(clf, prefit=True)
    return model.transform(X)
       
if __name__ == "__main__":
    thresh =(.01 * (1 - .01))
    # print(thresh)
    X = np.load('bined_x.npy')
    y = np.load('bined_y.npy')
    print('shapes of raw data',X.shape,y.shape)
    var_thresh_data = remove_with_var_thresh(X,thresh)
    print('shape after var thresh',var_thresh_data.shape)

    # uni_data = Univariate_feature_selection(X,y) #cannot work with negative data
    # print('shape of uni data',uni_data.shape)

    L1_data = L1_based_selection(X,y)
    print('shape of L1 data',L1_data.shape)

    tree_data = tree_selection(X,y)
    print('shape of tree data',tree_data.shape)

    # recursive_feature_elim(X,y) #plots stuff instead of returning