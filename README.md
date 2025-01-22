# hepatocytes-crypreservation-solutions

# Previsão da Taxa de Sobrevivência em Células Hepáticas

O **Previsor de Criopreservação** é um projeto da Fundação Oswaldo Cruz que utiliza técnicas de Machine Learning para prever a taxa de sobrevivência em células hepáticas submetidas a criopreservantes. O objetivo é desenvolver um modelo preditivo que possa ajudar a otimizar processos de criopreservação em aplicações biomédicas e tornar essa tecnologia de fácil acesso por meio de uma interface web.


## Instalação

Para instalar as dependências necessárias, você pode usar o `pip`. Execute o seguinte comando:

```bash
pip install -r requirements.txt
```

### Pontos importantes

- *Base de Dados:* Passei a utilizar as três tabelas "hepg1, mice e rat" como a fonte de dados.
- *Modelo:* Mantive o algorítimo genético do Vitor, mas apliquei à arquitetura Flask para funcionar no ambiente Web.
- *Ajustar de Filtro*: Filtrei todos os crioprotetores, para o resultado final ser uma métrica de quais crioprotetores resultam na melhor taxa de sobrevivência.
