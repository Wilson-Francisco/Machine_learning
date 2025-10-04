# importando as biblitecas
import pandas as pd
from sklearn import pipeline
from feature_engine.encoding import OneHotEncoder

# carregandos os dados
df = pd.read_csv("../data/Application_Data.csv")


features = ["Applicant_Gender",  "Income_Type","Education_Type","Family_Status", "Housing_Type", "Total_Income"]


# Separar as variaveis categoricas
list_var_categoricas = []
def var_categoricas():
    list_var_categoricas.clear()
    for i in df.columns[0:21].tolist():
        if df.dtypes[i] == 'object' or df.dtypes[i] == 'category':
            list_var_categoricas.append(i)


var_categoricas()

#onehor da lib features engine
onehot = OneHotEncoder(variables = list_var_categoricas)
onehot.fit(df[features])


# transformando o dado original
df_fit = onehot.transform(df[features])

# treinando o modelo de machine learning
target = "status"

clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(df_fit, df[target])


# Salvando o algoritmo
model = pd.Series(
    {
        "model": clf_tree,
        "onehot": onehot,
        "features": features,
        "tanget": target
    }
)
model.to_pickle("../models/aprov_credito_tree_onehot.pkl")

model