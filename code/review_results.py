from sklearn.metrics import (precision_recall_curve, average_precision_score, f1_score)
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import pandas as pd


class plot_results(object):
    
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        #y_pp = mod.predict_proba(X)[:, 1]
        self.y_pp = self.model.predict_proba(self.X_test)[:, 1]
        self.y_predict = self.model.predict(self.X_test)
        
    def plot_prauc(self):
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pp)
        avg_precision = average_precision_score(self.y_test, self.y_pp)
        fig, axr = plt.subplots(figsize=(16,10))

        axr.plot(recall, precision, label='PRAUC (area = %0.2f)' % avg_precision,
                 color='steelblue', linewidth=4,
                 alpha=0.7)

        axr.set_xlim([-0.05, 1.05])
        axr.set_ylim([0.0, 1.05])
        axr.set_xlabel('Recall', fontsize=16)
        axr.set_ylabel('Precision', fontsize=16)
        axr.set_title('Precision/Recall Curve\n', fontsize=20)

        axr.legend(loc="upper right", fontsize=12)

        path = './assets/precision_recall.png'
        plt.savefig(path, bbox_inches='tight')

        plt.show()

    def plot_roc(self):
        fpr_, tpr_, _ = roc_curve(self.y_test, self.y_pp)
        auc_ = auc(fpr_, tpr_)
        acc_ = np.abs(0.5 - np.mean(self.y_test)) + 0.5

        fig, axr = plt.subplots(figsize=(16,10))

        axr.plot(fpr_, tpr_, label='ROC (area = %0.2f)' % auc_,
                 color='darkred', linewidth=4,
                 alpha=0.7)
        axr.plot([0, 1], [0, 1], color='grey', ls='dashed',
                 alpha=0.9, linewidth=4)#, label='baseline accuracy = %0.2f' % acc_)
        axr.set_xlim([-0.05, 1.05])
        axr.set_ylim([0.0, 1.05])
        axr.set_xlabel('False Positive Rate', fontsize=16)
        axr.set_ylabel('True Positive Rate', fontsize=16)
        axr.set_title('ROC curve\n', fontsize=20)

        axr.legend(loc="lower right", fontsize=12)

        path = './assets/roc_curve.png'
        plt.savefig(path, bbox_inches='tight')

        plt.show()


    def _plot_confusion_matrix(self, cm, title='Confusion matrix', cmap=plt.cm.Blues, 
                              labels=('Funded Loan','Un-Funded Loan')):

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "%.4f" %cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
    def set_up_print_confusion_matrix(self):
            # Compute confusion matrix
        self.y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, self.y_pred)
        np.set_printoptions(precision=1)
        plt.figure()
        self._plot_confusion_matrix(cm)

        path = './assets/confusion_matrix.png'
        plt.savefig(path, bbox_inches='tight')
        # Normalize the confusion matrix by row (i.e by the number of samples
        # in each class)

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure()
        self._plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
        
        path = './assets/confusion_matrix_normalized.png'
        plt.savefig(path, bbox_inches='tight')
        
        plt.show()
            
    def print_classification_report(self):
        print classification_report(self.y_test, self.y_predict)
        
    def _print_top_words(self, model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
    
    def print_topic_model(self):
        cvec_feat = self.model.best_estimator_.get_params()['union'].get_params()['topic_model'].get_params()['CountVectorizer'].get_feature_names()
        lda_model = self.model.best_estimator_.get_params()['union'].get_params()['topic_model'].get_params()['lda']

        self._print_top_words(lda_model, cvec_feat, 10)

    def plot_feature_importance(self, n_features=25):

        a = self.model.best_estimator_.get_params()['union'].get_params()['non_text'].get_params()['GrabFeatures'].columns
        b = self.model.best_estimator_.get_params()['union'].get_params()['tf_idf'].get_params()['tfidf'].get_feature_names()
        c = len(self.model.best_estimator_.get_params()['union'].get_params()['topic_model'].get_params()['lda'].components_)
        c = ['Topic ' + str(n+1) for n in range(c)]

        model_cols = list(a) + list(b) + c

        coef = self.model.best_estimator_.get_params()['model'].coef_[0]
        features = pd.DataFrame({key:value for key,value in zip(model_cols,coef)},index=[1]).T.reset_index()
        features['coef(abs)'] = np.abs(features[1])
        features.rename(columns={'index':'Feature',1:'coef'},inplace=True)
        features.sort_values(by='coef(abs)',ascending=False, inplace=True)

        features.head(n_features).sort_values(by='coef').plot(kind='barh', x = 'Feature', y = 'coef',figsize=(16,8))
        plt.legend().set_visible(False)
        plt.xlabel("Coeffecient Value", fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.title('Top %s Features from Logistic Regressions'%(n_features), fontsize=20)

        path = './assets/feature_importance.png'
        plt.savefig(path, bbox_inches='tight')

        plt.show()