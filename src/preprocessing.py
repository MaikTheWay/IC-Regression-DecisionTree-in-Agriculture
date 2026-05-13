import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Classe para pré-processamento de dados seguindo padrões de engenharia de software.
    Utiliza Scikit-learn Pipelines para garantir que não haja vazamento de dados (data leakage).
    """
    def __init__(self, categorical_features, numerical_features, target_column, test_size=0.2, random_state=42):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None

    def create_pipeline(self):
        """Cria o transformador de colunas para variáveis numéricas e categóricas."""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        logger.info("Pipeline de pré-processamento criada.")
        return self.preprocessor

    def prepare_data(self, df: pd.DataFrame):
        """Separa features e target, e realiza o split treino/teste."""
        X = df[self.numerical_features + self.categorical_features]
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logger.info(f"Dados divididos: Treino={X_train.shape}, Teste={X_test.shape}")
        return X_train, X_test, y_train, y_test
