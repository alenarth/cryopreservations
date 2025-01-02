import pandas as pd
import os

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.data_path)
            self._preprocess_data()
        except Exception as e:
            print(f"Erro ao carregar os dados: {e}")

    def _preprocess_data(self):
        # Remover espaços em branco nas colunas
        self.data.columns = self.data.columns.str.strip()

        # Converter colunas de porcentagem em valores numéricos
        percent_cols = [
            '% DMSO', '% SFB', '% MEIO DE CULTURA', '% OUTRO CRIOPROTETOR',
            '% SOLUÇÃO TOTAL', '% ANTES DO CONGELAMENTO', '% APÓS O DESCONGELAMENTO'
        ]
        
        for col in percent_cols:
            self.data[col] = self.data[col].str.replace('%', '').str.replace(',', '.').astype(float)

        # Lidar com valores nulos
        self.data.fillna(0, inplace=True)

    def get_data(self):
        if self.data is None:
            self.load_data()
        return self.data

if __name__ == "__main__":
    data_loader = DataLoader('data/essays/hepg2.csv')
    data = data_loader.get_data()
    print(data.head())
