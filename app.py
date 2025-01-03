from flask import Flask, request, render_template
import torch
import numpy as np
import pandas as pd
from src.data_loader import DataLoader
from src.model import ModelTrainer

app = Flask(__name__)

# Carregar dados e treinar o modelo uma vez
print("Carregando dados e treinando modelo...")
data_loader = DataLoader('data/essays/hepg2.csv')
data = data_loader.get_data()
trainer = ModelTrainer(data)
trainer.train(epochs=700, lr=0.0005)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            dmso_value = float(request.form['dmso'])
            prediction = trainer.predict([dmso_value])
        except Exception as e:
            prediction = f"Erro: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)