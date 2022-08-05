#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import plot_confusion_matrix

from imblearn.pipeline import Pipeline as imbpipeline

from skopt import BayesSearchCV
from skopt.plots import plot_objective, plot_histogram

from timeit import default_timer as timer


## Bayesian search with k-fold cross validation

def bayes_kfold_pipeline(X_train, X_test, y_train, y_test, text_vec, scoring, \
                         classifier, parameters, cv_n_splits, bayes_n_iter, classifier_name, \
                         text_vec_name, resample_method_name, \
                         categories, n_classes, verbose, n_jobs,  resample):
    
    if resample == None:
        pipeline_model = Pipeline([text_vec, classifier])
    else:
        pipeline_model = imbpipeline([text_vec, resample, classifier])
        
     
    if cv_n_splits > 1:
        cv = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=12)
        search = BayesSearchCV(estimator=pipeline_model, search_spaces=parameters, n_iter=bayes_n_iter, \
                          scoring=scoring, cv=cv, n_jobs=n_jobs, \
                               verbose=verbose, random_state=12)
    elif cv_n_splits <=1:
        search = BayesSearchCV(estimator=pipeline_model, search_spaces=parameters, n_iter=bayes_n_iter, \
                          scoring=scoring, n_jobs=n_jobs, verbose=verbose, random_state=12)
        
    
    start = timer()
    
    search.fit(X_train, y_train)
    
    end = timer()
    print(f'{round(end - start)} seconds elapsed.')
    
    print("total iterations will be taken to explore all subspaces: ", search.total_iterations)
    
    print("------------------------------------------------------------------")
    print("\n Parameters of the best model:")
    best_params = search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_params[param_name]))

    
    print("------------------------------------------------------------------")
    

        
    y_pred_train = search.predict(X_train)
    y_pred = search.predict(X_test)
     
    conf_matrix_plot(search, X_train, y_train, X_test, y_test)
    
    print("Classification report for the train set:")
    print(classification_report(y_train, y_pred_train))
        
    print("Classification report for the test set:")
    print(classification_report(y_test, y_pred))
       
    print("------------------------------------------------------------------")
    
    prec0, recall0, fscore0, _ = precision_recall_fscore_support(y_test, y_pred, 
                                                  pos_label=0, average='binary')
    
    prec1, recall1, fscore1, _ = precision_recall_fscore_support(y_test, y_pred, 
                                                  pos_label=1, average='binary')
    
    roc_plot(search, X_test, y_test, categories, classifier_name, n_classes)
    
    prec_recall_plot(search, X_test, y_test, categories, classifier_name, n_classes)
    
    prec_recall_plot_multi(search, X_test, y_test, categories, classifier_name, n_classes)
    
        
    return text_vec_name, classifier_name, resample_method_name, \
            prec0, recall0, fscore0, prec1, recall1, fscore1
    

## ---------------------------------------------------------------------------------------------------
## ROC Curve
    
def roc_plot(estimator, X_test, y_test, categories, classifier_name, n_classes):
    
    categories = categories
    fig = plt.figure(figsize=(14,4))
    plt.rcParams.update({'font.size': 12})
    
    for i in n_classes:
        
        ax = fig.add_subplot(1, 2, i+1)
        
        display = RocCurveDisplay.from_estimator(estimator=estimator, X=X_test, y=y_test, pos_label=i, 
                                      response_method='predict_proba', ax=ax, name=classifier_name, 
                                                 color='r', lw=2)
        
        ax.plot([0,1], [0,1], color='navy', linestyle='--', lw=2)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Reciever Operating Characteristic (ROC) - {} '.format(categories[i]))

        
## ---------------------------------------------------------------------------------------------------
## Precision-Recall Curve

def prec_recall_plot(estimator, X_test, y_test, categories, classifier_name, n_classes):
    
    categories = categories
    fig = plt.figure(figsize=(14,4))
    plt.rcParams.update({'font.size': 12})
    
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    
    for i in n_classes:
        
        ax = fig.add_subplot(1, 2, i+1)
        
        display = PrecisionRecallDisplay.from_estimator(estimator=estimator, X=X_test, y=y_test, pos_label=i, 
                                      response_method='predict_proba', ax=ax, name=classifier_name,
                                                       color='g')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve - {} '.format(categories[i]))

        

## ---------------------------------------------------------------------------------------------------
## Precision-Recall and iso f1 Curve

def prec_recall_plot_multi(estimator, X_test, y_test, categories, classifier_name, n_classes):
    #n_classes = [0,1]
    
    color = ["teal", "deeppink"]
    categories = categories
    plt.rcParams.update({'font.size': 12})
    
    _, ax = plt.subplots(figsize=(8, 6))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.5)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
        
    
    for i in n_classes:
        
        display = PrecisionRecallDisplay.from_estimator(estimator=estimator, X=X_test, y=y_test, pos_label=i, 
                                      response_method='predict_proba', ax=ax,
                                      name='Precision-recall for {} class'.format(categories[i]),
                                                       color=color[i])
    ax.set_title('Precision-Recall Curve')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    ax.legend(handles=handles, labels=labels, loc="lower left")
    plt.show()

    
## ---------------------------------------------------------------------------------------------------
## Precision-Recall and iso f1 Curve

def conf_matrix_plot(model, X_train, y_train, X_test, y_test):
    
    fig = plt.figure(figsize=(14,4))
    
    ax1 = fig.add_subplot(1, 2, 1)
    plot_confusion_matrix(model, X_train, y_train, normalize='all', ax=ax1)
    ax1.set_title('Confusion matrix for the train set')
    
    ax2 = fig.add_subplot(1, 2, 2)
    plot_confusion_matrix(model, X_test, y_test, normalize='all', ax=ax2)
    ax2.set_title('Confusion matrix for the test set')