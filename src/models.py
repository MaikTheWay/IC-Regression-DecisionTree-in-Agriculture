from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Fábrica de modelos para encapsular a criação dos pipelines de machine learning.
    Garante que o pré-processamento seja aplicado consistentemente.
    """
    @staticmethod
    def create_linear_regression(preprocessor, params=None):
        if params is None:
            params = {}
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression(**params))
        ])
        logger.info("Modelo de Regressão Linear criado.")
        return model

    @staticmethod
    def create_decision_tree(preprocessor, params=None):
        if params is None:
            params = {'max_depth': 10, 'random_state': 42}
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor(**params))
        ])
        logger.info("Modelo de Árvore de Decisão criado.")
        return model
