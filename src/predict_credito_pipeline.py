import pandas as pd
from sklearn import pipeline
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from feature_engine.encoding import OneHotEncoder

# Carregandos os dados
df = pd.read_csv("../data/Application_Data.csv")

model = pd.read_pickle("../models/aprov_credito_tree_pipeline.pkl")

model

# Pegando infomacoes do usuario

features = [
    "Applicant_Gender",
    "Income_Type",
    "Total_Income" ,
    "Total_Bad_Debt",
    "Total_Good_Debt",
    "Education_Type",
    "Family_Status",
    "Job_Title",
    "Housing_Type"  ]

try:
    Applicant_Gender = str(input('Digite o seu Gênero: '))
    Income_Type = str(input('Digite o seu tipo de rendimento: '))
    Total_Income = int(input('Digite a sua renda Total: '))
    Total_Bad_Debt = int(input('Digite o seu total de dívida incobrável: '))
    Total_Good_Debt = int(input('Digite o seu total de dívida possíveis: '))
    Education_Type = str(input('Digite o seu tipo de escolaridade: '))
    Family_Status = str(input('Digite o seu estado Civil: '))
    Job_Title = str(input('Digite o seu cargo: '))
    Housing_Type = str(input('Digite o seu tipo de Habitação: '))

except ValueError:

    print("Por favor, digite apenas números!")

data = pd.DataFrame(
    {
        "Applicant_Gender": [Applicant_Gender],
        "Income_Type" : [Income_Type],
        "Total_Income" : [Total_Income],
        "Total_Bad_Debt" : [Total_Bad_Debt],
        "Total_Good_Debt" : [Total_Good_Debt],
        "Education_Type" : [Education_Type],
        "Family_Status" : [Family_Status],
        "Job_Title" : [Job_Title],
        "Housing_Type": [Housing_Type]
    } )

pred = model["model"].predict(data[model["features"]])[0]
print(f'O seu credito foi: {pred}')