import torch
import pandas as pd
from data_loader import DataLoader
from model import ModelTrainer

if __name__ == "__main__":
    # Carregar dados
    data_loader = DataLoader('data/essays/hepg2.csv')
    data = data_loader.get_data()

    # Treinar o modelo
    trainer = ModelTrainer(data)
    trainer.train(epochs=300, lr=0.001)

    # Solicitar entrada do usuário
    while True:
        try:
            input_value = input("Digite a porcentagem de DMSO que deseja usar: ")
            X = [float(input_value)]

            prediction = trainer.predict(X)
            print(f"Chance de sobrevivência: {prediction:.2f}%")

        except Exception as e:
            print(f"Erro ao processar a entrada: {e}")

        continuar = input("Deseja fazer outra previsão? (s/n): ")
        if continuar.lower() != 's':
            break
