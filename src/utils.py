import logging
import os
import yaml
import json
import pandas as pd

def setup_logging(log_dir="logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "project.log")),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_results(results, output_dir="outputs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Salvar em JSON
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Salvar em CSV para fácil visualização
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    
    print("\n" + "="*50)
    print("RELATÓRIO COMPARATIVO")
    print("="*50)
    print(df_results.to_string(index=False))
    print("="*50)
    print(f"Resultados salvos em: {output_dir}")
