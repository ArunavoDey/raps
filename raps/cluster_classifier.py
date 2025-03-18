import joblib
import os
from pathlib import Path
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError
from .cluster_classifier_factory import ClusterClassifierFactory

class ClusterClassifier:
    """Handles classification of data into pre-defined clusters"""

    _Classifiers = {
            'random_forest': ClusterClassifierFactory._create_random_forest,
            'logistic_regression': ClusterClassifierFactory._create_logistic_regression,
            #'xgboost': ClusterClassifierFactory._create_xgboost
        }
    
    def __init__(self, dataLoader):
        """
        Initialize cluster classifier
        Args:
            dataLoader: dataloader
        """
        self.dataLoader = dataLoader

        self.classifier = "random_forest" #dataLoader.config.get('datasets',{}).get(dataLoader.dataset_name, {}).get('cluster_classifier', {}).get('algorithm', 'random_forest')
        if self.classifier not in self._Classifiers:
            raise ValueError(f"Unsupported Classifier: {self.classifier}")
        
        self.model=None
        self.feature_map = None

    def train(self, data, clusters):
        try:
            self.load(self.dataLoader)
        except FileNotFoundError:
            print(f"No existing classifier found. Training new {self.classifier} model.")
        
            X_features = self.dataLoader.config.get('datasets',{}).get(self.dataLoader.dataset_name, {}).get('data', {}).get('X_features', '')
            X_common = data[X_features]
            # Create and train cluster classifier
            self.fit(X_common, clusters)
            self.save()

    def fit(self, X, cluster_labels):
        """
        Train cluster classifier
        Args:
            X: Feature matrix (common features only)
            cluster_labels: Cluster assignments from training
        """
        self.feature_map = list(X.columns)
        params = self.dataLoader.config.get('datasets',{}).get(self.dataLoader.dataset_name, {}).get('cluster_classifier', {}).get('params', '')

        self.model = self._Classifiers[self.classifier](params)
        self.model.fit(X, cluster_labels)

    def save(self) -> None:
        """Save classifier and metadata to disk"""
        if self.model is None:
            raise RuntimeError("No model to save. Fit first.")
        
        model_path = f"models/cluster_classifiers/{self.dataLoader.dataset_name}/{self.classifier}.pkl"
        Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_map': self.feature_map
        }, model_path)

    
    @staticmethod
    def _predict_clusters(dataLoader, processed_data):
        """Predict cluster assignments for test data"""

         # Load cluster classifier
        cluster_classifier = ClusterClassifier.load(dataLoader)        
        
        if not cluster_classifier:
            raise NotFittedError("Cluster classifier not loaded")
        return cluster_classifier.predict(processed_data)

    @classmethod
    def load(cls, dataLoader):
        """Load saved classifier from disk"""

        classifier = "random_forest" #dataLoader.config.get('cluster_classifier', {}).get('algorithm', 'random_forest')
        model_path = f"/work/08389/hcs77/ls6/power-prediction/models/cluster_classifiers/{dataLoader.dataset_name}/{classifier}.pkl"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No cluster classifier found at {model_path}")
            
        data = joblib.load(model_path)
        instance = cls(dataLoader)
        instance.model = data['model']
        instance.feature_map = data['feature_map']
        return instance

    def predict(self, X):
        """
        Predict cluster assignments for new data
        Args:
            X: Test data with common features
        Returns:
            Array of cluster assignments
        """
        if self.model is None:
            raise NotFittedError("Classifier not trained. Call fit() first.")
        
        aligned_X = self._align_features(X)
        predicted_clusters = self.model.predict(aligned_X)
        probabilities = self.model.predict_proba(aligned_X)
        return predicted_clusters, probabilities
    

    def _align_features(self, X):
        """Ensure test data matches training feature space"""
        return X.reindex(columns=self.feature_map, fill_value=0)
