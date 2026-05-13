# IC - Comparação entre Regressão Linear e Árvore de Decisão na Agricultura de Precisão

Este projeto faz parte de uma Iniciação Científica (IC) vinculada à Unimetrocamp Wyden. O objetivo é comparar o desempenho de dois algoritmos de Machine Learning na previsão de produtividade agrícola (`hg/ha_yield`).

## Estrutura do Projeto

- `src/`: Módulos Python (Carregamento, Pré-processamento, Modelagem, Avaliação).
- `data/`: Dataset original (`yield_df.csv`).
- `configs/`: Arquivo de configuração YAML.
- `outputs/`: Resultados do experimento (métricas em CSV e JSON).
- `logs/`: Logs de execução para rastreabilidade acadêmica.
- `main.py`: Ponto de entrada do sistema.

## Como Executar

1. **Configurar o Ambiente**:
   - Dependências listadas em `environment.yml`

   ```bash
   micromamba env create -f environment.yml
   micromamba activate agri_ic
   ```

2. **Rodar o Experimento**:
   ```bash
   python main.py
   ```

3. **Parâmetros Customizados**:
   ```bash
   python main.py --config configs/config.yaml --data data/yield_df.csv
   ```

---
**Orientador:** Eder Carlos Fernandes
**Pesquisador:** Marcos Alcino Ribeiro Cussioli
**Instituição:** Unimetrocamp Wyden
