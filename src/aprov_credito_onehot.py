# importando as biblitecas
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from feature_engine.encoding import OneHotEncoder

# Carregandos os dados
df = pd.read_csv("../data/Application_Data.csv")


features = ['Income_Type', 'Family_Status', 'Housing_Type', 'Job_Title', 'Total_Income', 'Total_Bad_Debt', 'Total_Good_Debt']


# Separar as variaveis categoricas
list_var_categoricas = []
def var_categoricas():
    list_var_categoricas.clear()
    for i in df.columns[0:21].tolist():
        if df.dtypes[i] == 'object' or df.dtypes[i] == 'category':
            list_var_categoricas.append(i)

# Chamando a funcao
var_categoricas()

#Onehot da lib features engine
onehot = OneHotEncoder(variables = list_var_categoricas)
onehot.fit(df[features])

# Transformando o dado original
df_fit = onehot.transform(df[features])

# Treinando o modelo de machine learning
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