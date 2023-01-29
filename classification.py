import numpy,scipy
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

def get_cross_validation(data,labels):

    names = [
    #"Nearest Neighbors", 
    #"Neural Net",
    "RBF SVM",
    #"Decision Tree",
    #"Random Forest",
    #"AdaBoost" 
    ]
    
    classifiers = [
        #KNeighborsClassifier(3),
        #MLPClassifier(alpha=1),
        SVC(gamma=0.01, C=100),
        #DecisionTreeClassifier(max_depth=5),
        #RandomForestClassifier(max_depth=5, n_estimators=1000, max_features=1),
        #AdaBoostClassifier()
        ]

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        print(name)
        #kf1 = KFold(n_splits=4)
        #kf2 = KFold(n_splits=5)
        #kf3 = KFold(n_splits=7)
        #kf4 = KFold(n_splits=9)
        kf5 = KFold(n_splits=10)
        #score1 = cross_val_score(clf, data, labels, cv=kf1)
        #score2 = cross_val_score(clf, data, labels, cv=kf2)
        #score3 = cross_val_score(clf, data, labels, cv=kf3)
        #score4 = cross_val_score(clf, data, labels, cv=kf4)
        score5 = cross_val_score(clf, data, labels, cv=kf5)
        print("classifier ==>", name,  " 10 cross validation acc == ",numpy.mean(score5))
        
        
        
        
        
        '''
        scoring = ['accuracy', 'precision_weighted','recall_weighted','f1_weighted']
        kfold = KFold(n_splits=4, random_state=42)
    

        results = cross_validate(estimator=clf,
                                          X=data,
                                          y=labels,
                                          cv=kfold,
                                          scoring=scoring)
        print(results.keys())
        avg_acc = numpy.mean(results['test_accuracy'])
        avg_prec = numpy.mean(results['test_precision_weighted'])
        avg_recall = numpy.mean(results['test_recall_weighted'])
        avg_fscore = numpy.mean(results['test_f1_weighted'])

                                         
        print("classifier ==>", name,  " acc == ",avg_acc)
        print("classifier ==>", name,  " precision == ",avg_prec)
        print("classifier ==>", name,  " recall == ",avg_recall)
        print("classifier ==>", name,  " f-score == ",avg_fscore)
        
        '''
        # shuffle and split training and test sets
        #X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.3,random_state=42)
        # fit a model
        #clf.fit(X_train, y_train)
        #y_pred=clf.predict(X_test)
        #print(classification_report(y_test, y_pred))

        # predict probabilities
        # keep probabilities for the positive outcome only
        #probs = probs[:, 1]
        # calculate AUC
        #auc = roc_auc_score(y_test, probs)
        #print('AUC: %.3f' % auc)
        # calculate roc curve
        #fpr, tpr, thresholds = roc_curve(y_test, probs)
        # plot no skill
        #pyplot.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        #pyplot.plot(fpr, tpr, marker='.')
        # show the plot
        #pyplot.show()

        # Calculate precision and recall from true labels vs score values
        #precision, recall, _ = precision_recall_curve(y_true, y_score)
        


    
