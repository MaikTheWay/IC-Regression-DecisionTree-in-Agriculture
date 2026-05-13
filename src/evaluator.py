import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import logging
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Classe para avaliação de modelos de regressão.
    Inclui métricas padrão, validação cruzada e análise de importância de variáveis.
    """
    def __init__(self, X_test, y_test, feature_names=None):
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names

    def evaluate(self, model, model_name: str):
        """Calcula R2, MAE e RMSE para o conjunto de teste."""
        predictions = model.predict(self.X_test)
        
        r2 = r2_score(self.y_test, predictions)
        mae = mean_absolute_error(self.y_test, predictions)
        mse = mean_squared_error(self.y_test, predictions)
        rmse = np.sqrt(mse)

        metrics = {
            'Model': model_name,
            'R2': float(r2),
            'MAE': float(mae),
            'RMSE': float(rmse)
        }
        
        logger.info(f"Métricas para {model_name}: {metrics}")
        return metrics

    def cross_validate(self, model, X, y, cv=5):
        """Realiza validação cruzada para verificar estabilidade do modelo."""
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        result = {
            'CV_R2_Mean': float(scores.mean()),
            'CV_R2_Std': float(scores.std())
        }
        logger.info(f"Validação Cruzada ({cv} folds) - Média: {result['CV_R2_Mean']:.4f}, Std: {result['CV_R2_Std']:.4f}")
        return result

    def get_feature_importance(self, model, model_name: str, output_dir="outputs"):
        """Extrai importância das variáveis ou coeficientes conforme o modelo."""
        regressor = model.named_steps['regressor']
        preprocessor = model.named_steps['preprocessor']
        
        # Tentar obter nomes das colunas após o OneHotEncoding
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = self.feature_names

        importance_df = None
        
        if hasattr(regressor, 'feature_importances_'):
            importance = regressor.feature_importances_
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
        elif hasattr(regressor, 'coef_'):
            importance = regressor.coef_
            importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': importance})
            importance_df = importance_df.sort_values(by='Coefficient', key=abs, ascending=False)

        if importance_df is not None:
            path = os.path.join(output_dir, f"importance_{model_name.lower().replace(' ', '_')}.csv")
            importance_df.to_csv(path, index=False)
            logger.info(f"Importância das variáveis para {model_name} salva em {path}")
            
        return importance_df

    def plot_residuals(self, model, model_name: str, output_dir="outputs"):
        """Gera gráfico de resíduos para análise de erro."""
        predictions = model.predict(self.X_test)
        residuals = self.y_test - predictions
        
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Previsões')
        plt.ylabel('Resíduos')
        plt.title(f'Gráfico de Resíduos - {model_name}')
        
        path = os.path.join(output_dir, f"residuals_{model_name.lower().replace(' ', '_')}.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Gráfico de resíduos para {model_name} salvo em {path}")

    def plot_prediction_vs_actual(self, model, model_name: str, output_dir="outputs"):
        """Gera gráfico de Previsão vs Valor Real."""
        predictions = model.predict(self.X_test)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, predictions, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Valores Reais')
        plt.ylabel('Previsões')
        plt.title(f'Previsão vs Real - {model_name}')
        
        path = os.path.join(output_dir, f"prediction_vs_actual_{model_name.lower().replace(' ', '_')}.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Gráfico Previsão vs Real para {model_name} salvo em {path}")

    def plot_error_distribution(self, model, model_name: str, output_dir="outputs"):
        """Gera histograma da distribuição dos erros."""
        predictions = model.predict(self.X_test)
        errors = self.y_test - predictions
        
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Erro de Previsão')
        plt.ylabel('Frequência')
        plt.title(f'Distribuição de Erros - {model_name}')
        
        path = os.path.join(output_dir, f"error_distribution_{model_name.lower().replace(' ', '_')}.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Gráfico de distribuição de erros para {model_name} salvo em {path}")

    @staticmethod
    def plot_model_comparison(results, output_dir="outputs"):
        """Gera gráficos comparativos entre os modelos."""
        df = pd.DataFrame(results)
        
        # Plot R2 Comparison
        plt.figure(figsize=(10, 6))
        plt.bar(df['Model'], df['R2'], color=['#3498db', '#2ecc71'])
        plt.ylabel('R² Score')
        plt.title('Comparação de Desempenho (R²)')
        plt.ylim(0, 1.1)
        for i, v in enumerate(df['R2']):
            plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
        
        plt.savefig(os.path.join(output_dir, "comparison_r2.png"))
        plt.close()

        # Plot MAE/RMSE Comparison
        metrics_to_plot = ['MAE', 'RMSE']
        df_melted = df.melt(id_vars='Model', value_vars=metrics_to_plot, var_name='Métrica', value_name='Valor')
        
        import seaborn as sns
        plt.figure(figsize=(12, 7))
        sns.barplot(data=df_melted, x='Métrica', y='Valor', hue='Model', palette='viridis')
        plt.title('Comparação de Erros (MAE e RMSE)')
        plt.savefig(os.path.join(output_dir, "comparison_errors.png"))
        plt.close()
        
        logger.info(f"Gráficos comparativos de modelos salvos em {output_dir}")
