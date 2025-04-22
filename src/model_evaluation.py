from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc
)
import pandas as pd
import shap

class ModelEvaluation:
    def __init__(self, model, x_test, y_test,
                        x_train, y_train) -> None:
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.explainer = None
        self.shap_values = None
        self.pred_test = None
        
    def get_model_auc_metric(self):
        """
        Get the area under the ROC curve for train and test set. 
        Args:
            model (any): model to evaluate
            x_test (DataFrame): test set features
            y_test (Series): test labels 
            x_train (DataFrame): train set features
            y_train (Series): train labels
        """
        self.pred_test = pd.Series(self.model.predict_proba(self.x_test)[:, 1])
        self.pred_test.index = self.x_test.index
        test_auc = roc_auc_score(self.y_test, self.pred_test)
        pred_train = pd.Series(self.model.predict_proba(self.x_train)[:, 1])
        pred_train.index = self.x_train.index
        train_auc = roc_auc_score(self.y_train, pred_train)
        print(f"Train set AUC:{train_auc} \n")
        print(f"Test set AUC: {test_auc}")

    def get_model_aucp(self):
        """
        Get the area under the Precision-Recall curve.

        Args:
            model (any): model to evaluate. 
            x_test (DataFrame): test set features
            y_test (Series): test labels.
        """
        precision, recall, _ = precision_recall_curve(self.y_test,
                                                      self.pred_test)
        aucpr = auc(recall, precision)
        print(f"Test set AUCPR: {aucpr}")

    def get_shap_values(self):
        self.explainer = shap.TreeExplainer(self.model,
                                            self.x_test)
        self.shap_values = self.explainer.shap_values(
                                            self.x_test
                                        )

    def get_shap_summary_plot(self):
        shap.summary_plot(self.shap_values,
                          self.x_test)

    def get_shap_decision_plot(self, response):
        index = 257182
        outage_id = "99962_34029_2015100113"
        if response == 0:
            index = 3
            outage_id = "100009_55009_2015081417"
        shap.decision_plot(self.explainer.expected_value,
                            self.shap_values[index,:],
                            self.x_test.loc[outage_id,:],
                            link='logit')
