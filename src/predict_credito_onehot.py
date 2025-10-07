import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from feature_engine.encoding import OneHotEncoder

# Carregandos os dados
df = pd.read_csv("../data/Application_Data.csv")

# Carreagando  modelo
model = pd.read_pickle("../models/aprov_credito_tree_onehot.pkl")

model


# Pegando informações do usuario

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

# Transformando os inputs do usuario em dataframe
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

# Fazendo as predição dos inputs do usuario
df_full = model['onehot'].transform(data)
pred = model["model"].predict(df_full)[0]

print(f'O seu credito foi: {pred}')
#proba = model["model"].predict_proba(df_final(model['features']))