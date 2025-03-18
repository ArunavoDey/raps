from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier
from typing import Any, Dict

class ClusterClassifierFactory:
    """Factory for creating cluster classification models"""

    @staticmethod
    def _create_random_forest(params):
        required = ['n_estimators']
        ClusterClassifierFactory._validate_params(params, required)
        return RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params.get('max_depth', None),
            random_state=params.get('random_state', None),
            n_jobs=params.get('n_jobs', -1)
        )

    @staticmethod
    def _create_logistic_regression(params):
        return LogisticRegression(
            C=params.get('C', 1.0),
            max_iter=params.get('max_iter', 100),
            random_state=params.get('random_state', None)
        )

    """
    @staticmethod
    def _create_xgboost(params):
        required = ['n_estimators']
        ClusterClassifierFactory._validate_params(params, required)
        return XGBClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params.get('learning_rate', 0.1),
            max_depth=params.get('max_depth', 3),
            random_state=params.get('random_state', None)
        )
    """

    @staticmethod
    def _validate_params(params, required):
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")