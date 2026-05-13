import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Classe responsável pelo carregamento e validação básica dos dados agrícolas.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.file_path):
            logger.error(f"Arquivo não encontrado: {self.file_path}")
            raise FileNotFoundError(f"O dataset {self.file_path} é obrigatório para a execução.")
        
        try:
            # O dataset yield_df.csv possui uma coluna de índice sem nome no início
            df = pd.read_csv(self.file_path)
            # Remover coluna de índice se existir
            if df.columns[0].startswith('Unnamed') or df.columns[0] == '':
                df = df.iloc[:, 1:]
            
            logger.info(f"Dados carregados com sucesso. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Erro ao ler o CSV: {e}")
            raise

    def validate_columns(self, df: pd.DataFrame, required_columns: list):
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.error(f"Colunas ausentes no dataset: {missing}")
            raise ValueError(f"O dataset não contém as colunas necessárias: {missing}")
        logger.info("Validação de colunas concluída.")
