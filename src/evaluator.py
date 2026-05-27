import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Classe para avaliação de modelos de regressão.
    Inclui métricas padrão, validação cruzada, análise de importância de variáveis,
    diagnóstico de resíduos (Regressão Linear) e busca empírica de profundidade
    (Árvore de Decisão).
    """

    def __init__(self, X_test, y_test, feature_names=None):
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names

    # ------------------------------------------------------------------
    # Métricas de avaliação
    # ------------------------------------------------------------------

    def evaluate(self, model, model_name: str) -> dict:
        """Calcula R², MAE e RMSE para o conjunto de teste."""
        predictions = model.predict(self.X_test)

        r2   = r2_score(self.y_test, predictions)
        mae  = mean_absolute_error(self.y_test, predictions)
        mse  = mean_squared_error(self.y_test, predictions)
        rmse = np.sqrt(mse)

        metrics = {
            'Model': model_name,
            'R2':    float(r2),
            'MAE':   float(mae),
            'RMSE':  float(rmse)
        }

        logger.info(f"Métricas para {model_name}: {metrics}")
        return metrics

    def cross_validate(self, model, X, y, cv: int = 5) -> dict:
        """Realiza validação cruzada para verificar estabilidade do modelo."""
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        result = {
            'CV_R2_Mean': float(scores.mean()),
            'CV_R2_Std':  float(scores.std())
        }
        logger.info(
            f"Validação Cruzada ({cv} folds) — "
            f"Média: {result['CV_R2_Mean']:.4f}, Std: {result['CV_R2_Std']:.4f}"
        )
        return result

    # ------------------------------------------------------------------
    # Importância / coeficientes
    # ------------------------------------------------------------------

    def get_feature_importance(self, model, model_name: str, output_dir: str = "outputs"):
        """Extrai importância das variáveis ou coeficientes conforme o modelo."""
        regressor   = model.named_steps['regressor']
        preprocessor = model.named_steps['preprocessor']

        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = self.feature_names

        importance_df = None

        if hasattr(regressor, 'feature_importances_'):
            importance    = regressor.feature_importances_
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
        elif hasattr(regressor, 'coef_'):
            importance    = regressor.coef_
            importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': importance})
            importance_df = importance_df.sort_values(by='Coefficient', key=abs, ascending=False)

        if importance_df is not None:
            safe_name = model_name.lower().replace(' ', '_')
            path = os.path.join(output_dir, f"importance_{safe_name}.csv")
            importance_df.to_csv(path, index=False)
            logger.info(f"Importância das variáveis para {model_name} salva em {path}")

        return importance_df

    # ------------------------------------------------------------------
    # Gráficos de diagnóstico — padrão
    # ------------------------------------------------------------------

    def plot_residuals(self, model, model_name: str, output_dir: str = "outputs"):
        """Gera gráfico de resíduos (resíduos vs. previsões)."""
        predictions = model.predict(self.X_test)
        residuals   = self.y_test - predictions

        plt.figure(figsize=(10, 6))
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Previsões')
        plt.ylabel('Resíduos')
        plt.title(f'Gráfico de Resíduos — {model_name}')

        safe_name = model_name.lower().replace(' ', '_')
        path = os.path.join(output_dir, f"residuals_{safe_name}.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        logger.info(f"Gráfico de resíduos para {model_name} salvo em {path}")

    def plot_prediction_vs_actual(self, model, model_name: str, output_dir: str = "outputs"):
        """Gera gráfico de Previsão vs. Valor Real."""
        predictions = model.predict(self.X_test)

        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, predictions, alpha=0.5)
        plt.plot(
            [self.y_test.min(), self.y_test.max()],
            [self.y_test.min(), self.y_test.max()],
            'r--', lw=2
        )
        plt.xlabel('Valores Reais')
        plt.ylabel('Previsões')
        plt.title(f'Previsão vs. Real — {model_name}')

        safe_name = model_name.lower().replace(' ', '_')
        path = os.path.join(output_dir, f"prediction_vs_actual_{safe_name}.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        logger.info(f"Gráfico Previsão vs. Real para {model_name} salvo em {path}")

    def plot_error_distribution(self, model, model_name: str, output_dir: str = "outputs"):
        """Gera histograma da distribuição dos erros."""
        predictions = model.predict(self.X_test)
        errors      = self.y_test - predictions

        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Erro de Previsão')
        plt.ylabel('Frequência')
        plt.title(f'Distribuição de Erros — {model_name}')

        safe_name = model_name.lower().replace(' ', '_')
        path = os.path.join(output_dir, f"error_distribution_{safe_name}.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        logger.info(f"Gráfico de distribuição de erros para {model_name} salvo em {path}")

    # ------------------------------------------------------------------
    # Diagnóstico de premissas — Regressão Linear (NOVO)
    # ------------------------------------------------------------------

    def plot_linear_regression_diagnostics(
        self, model, output_dir: str = "outputs"
    ):
        """
        Gera dois gráficos de diagnóstico para a Regressão Linear:
          1. Resíduos vs. Valores Previstos — verifica padrões sistemáticos
             que indicam violação da premissa de linearidade.
          2. Q-Q Plot dos resíduos — verifica a premissa de normalidade.

        Esses gráficos foram adicionados em resposta à revisão acadêmica que
        apontou a necessidade de evidência empírica de não-linearidade nos dados
        (Melhoria #3 do relatório de revisão).
        """
        predictions = model.predict(self.X_test)
        residuals   = self.y_test - predictions

        # --- Gráfico 1: Resíduos vs. Valores Previstos ---
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions, residuals, alpha=0.4, color='steelblue')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
        plt.xlabel('Valores Previstos (hg/ha)')
        plt.ylabel('Resíduos (hg/ha)')
        plt.title('Resíduos vs. Valores Previstos — Regressão Linear\n'
                  '(Padrão sistemático indica violação da premissa de linearidade)')
        plt.tight_layout()
        path_resid = os.path.join(output_dir, "lr_residuals_vs_predicted.png")
        plt.savefig(path_resid, bbox_inches='tight', dpi=150)
        plt.close()
        logger.info(f"Diagnóstico de resíduos (RL) salvo em {path_resid}")

        # --- Gráfico 2: Q-Q Plot ---
        plt.figure(figsize=(8, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot dos Resíduos — Regressão Linear\n'
                  '(Desvio da diagonal indica violação da premissa de normalidade)')
        plt.tight_layout()
        path_qq = os.path.join(output_dir, "lr_qq_plot.png")
        plt.savefig(path_qq, bbox_inches='tight', dpi=150)
        plt.close()
        logger.info(f"Q-Q Plot dos resíduos (RL) salvo em {path_qq}")

        return path_resid, path_qq

    # ------------------------------------------------------------------
    # Busca empírica de profundidade — Árvore de Decisão (NOVO)
    # ------------------------------------------------------------------

    def run_depth_sweep(
        self,
        preprocessor,
        X_train,
        y_train,
        depths: list = None,
        output_dir: str = "outputs",
        random_state: int = 42,
        cv: int = 5
    ) -> pd.DataFrame:
        """
        Realiza uma busca empírica de profundidade máxima para a Árvore de Decisão,
        calculando R² em treino e em validação cruzada para cada valor de max_depth.

        Essa análise foi adicionada em resposta à revisão acadêmica que apontou
        a falta de justificativa empírica para max_depth=10 (Melhoria #2 do
        relatório de revisão).

        Parâmetros
        ----------
        preprocessor : sklearn ColumnTransformer
            Pré-processador já configurado (não fitado).
        X_train, y_train : array-like
            Conjunto de treino.
        depths : list, opcional
            Lista de valores de max_depth a testar. Padrão: [3, 5, 10, 15, 20].
        output_dir : str
            Diretório para salvar CSV e gráfico.
        random_state : int
            Semente aleatória para reprodutibilidade.
        cv : int
            Número de folds para validação cruzada.

        Retorna
        -------
        pd.DataFrame com colunas: max_depth, R2_train, R2_cv_mean, R2_cv_std, diff_overfit
        """
        if depths is None:
            depths = [3, 5, 10, 15, 20]

        results = []
        for d in depths:
            dt_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', DecisionTreeRegressor(max_depth=d, random_state=random_state))
            ])

            # Validação cruzada (usa clone interno do pipeline)
            cv_scores = cross_val_score(dt_pipeline, X_train, y_train, cv=cv, scoring='r2')

            # Treino completo para R² de treino
            dt_pipeline.fit(X_train, y_train)
            train_score = dt_pipeline.score(X_train, y_train)

            results.append({
                'max_depth':    d,
                'R2_train':     round(float(train_score), 4),
                'R2_cv_mean':   round(float(np.mean(cv_scores)), 4),
                'R2_cv_std':    round(float(np.std(cv_scores)), 4),
                'diff_overfit': round(float(train_score - np.mean(cv_scores)), 4)
            })
            logger.info(
                f"Depth={d}: R²_train={train_score:.4f}, "
                f"R²_cv={np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}, "
                f"diff={train_score - np.mean(cv_scores):.4f}"
            )

        results_df = pd.DataFrame(results)

        # Salvar CSV
        csv_path = os.path.join(output_dir, "dt_depth_sweep.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Resultados da busca de profundidade salvos em {csv_path}")

        # Gráfico
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['max_depth'], results_df['R2_train'],
                 label='R² Treino', marker='o', color='tomato')
        plt.plot(results_df['max_depth'], results_df['R2_cv_mean'],
                 label='R² Validação Cruzada (média)', marker='s', color='steelblue')
        plt.fill_between(
            results_df['max_depth'],
            results_df['R2_cv_mean'] - results_df['R2_cv_std'],
            results_df['R2_cv_mean'] + results_df['R2_cv_std'],
            alpha=0.2, color='steelblue', label='±1 Desvio Padrão (CV)'
        )
        plt.axvline(x=10, color='green', linestyle=':', linewidth=1.5,
                    label='max_depth=10 (escolhido)')
        plt.xlabel('Profundidade Máxima (max_depth)')
        plt.ylabel('R²')
        plt.title('Desempenho da Árvore de Decisão por Profundidade\n'
                  '(Busca empírica com validação cruzada de 5 folds)')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "dt_depth_sweep.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        logger.info(f"Gráfico de busca de profundidade salvo em {plot_path}")

        return results_df

    # ------------------------------------------------------------------
    # Comparação global
    # ------------------------------------------------------------------

    @staticmethod
    def plot_model_comparison(results: list, output_dir: str = "outputs"):
        """Gera gráficos comparativos entre os modelos."""
        import seaborn as sns

        df = pd.DataFrame(results)

        # R² Comparison
        plt.figure(figsize=(10, 6))
        plt.bar(df['Model'], df['R2'], color=['#3498db', '#2ecc71'])
        plt.ylabel('R² Score')
        plt.title('Comparação de Desempenho (R²)')
        plt.ylim(0, 1.1)
        for i, v in enumerate(df['R2']):
            plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
        plt.savefig(os.path.join(output_dir, "comparison_r2.png"), bbox_inches='tight')
        plt.close()

        # MAE / RMSE Comparison
        metrics_to_plot = ['MAE', 'RMSE']
        df_melted = df.melt(
            id_vars='Model', value_vars=metrics_to_plot,
            var_name='Métrica', value_name='Valor'
        )
        plt.figure(figsize=(12, 7))
        sns.barplot(data=df_melted, x='Métrica', y='Valor', hue='Model', palette='viridis')
        plt.title('Comparação de Erros (MAE e RMSE)')
        plt.savefig(os.path.join(output_dir, "comparison_errors.png"), bbox_inches='tight')
        plt.close()

        logger.info(f"Gráficos comparativos de modelos salvos em {output_dir}")
