import argparse
import logging
import sys
import os
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.models import ModelFactory
from src.evaluator import Evaluator
from src.utils import setup_logging, load_config, save_results

def main():
    parser = argparse.ArgumentParser(description="IC - Comparação Regressão Linear vs Árvore de Decisão")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Caminho para o arquivo de config")
    parser.add_argument("--data", type=str, help="Caminho para o dataset (sobrescreve config)")
    args = parser.parse_args()

    # Configuração inicial
    config = load_config(args.config)
    setup_logging(config['output']['log_dir'])
    logger = logging.getLogger(__name__)
    
    output_dir = config['output']['dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Iniciando Experimento!")

    try:
        # 1. Carregamento
        data_path = args.data if args.data else config['data']['path']
        loader = DataLoader(data_path)
        df = loader.load_data()
        
        required_cols = config['data']['categorical_features'] + \
                        config['data']['numerical_features'] + \
                        [config['project']['target_column']]
        loader.validate_columns(df, required_cols)

        # 2. Pré-processamento
        preprocessor_obj = Preprocessor(
            categorical_features=config['data']['categorical_features'],
            numerical_features=config['data']['numerical_features'],
            target_column=config['project']['target_column'],
            test_size=config['project']['test_size'],
            random_state=config['project']['random_state']
        )
        X_train, X_test, y_train, y_test = preprocessor_obj.prepare_data(df)
        pipeline = preprocessor_obj.create_pipeline()

        # 3. Modelagem e Treinamento
        models = {
            "Regressão Linear": ModelFactory.create_linear_regression(
                pipeline, config['models']['linear_regression']['params']
            ),
            "Árvore de Decisão": ModelFactory.create_decision_tree(
                pipeline, config['models']['decision_tree']['params']
            )
        }

        results = []
        evaluator = Evaluator(X_test, y_test)

        for name, model in models.items():
            logger.info(f"Processando modelo: {name}")
            
            # Treino
            model.fit(X_train, y_train)
            
            # Avaliação Simples
            metrics = evaluator.evaluate(model, name)
            
            # Validação Cruzada
            logger.info(f"Iniciando Validação Cruzada para {name}...")
            cv_results = evaluator.cross_validate(model, X_train, y_train)
            metrics.update(cv_results)
            
            # Interpretabilidade
            evaluator.get_feature_importance(model, name, output_dir)
            
            # Visualização
            evaluator.plot_residuals(model, name, output_dir)
            evaluator.plot_prediction_vs_actual(model, name, output_dir)
            evaluator.plot_error_distribution(model, name, output_dir)
            
            results.append(metrics)

        # 4. Comparação Global e Exportação Final
        evaluator.plot_model_comparison(results, output_dir)
        save_results(results, output_dir)
        logger.info("Experimento concluído :)")

    except Exception as e:
        logger.error(f"Falha na execução: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
