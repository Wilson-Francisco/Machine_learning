# Importando as biblitecas
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from feature_engine.encoding import OneHotEncoder

# Carregandos os dados
df = pd.read_csv("../data/Application_Data.csv")


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


# Renomeando a coluna de Status
df['Status'].replace( {0: "Credit not approved", 1: "Credit approved"}, inplace = True)


# Separar as variaveis categoricas
list_var_categoricas = []
def var_categoricas():
    list_var_categoricas.clear()
    for i in df.columns[0:21].tolist():
        if df.dtypes[i] == 'object' or df.dtypes[i] == 'category':
            list_var_categoricas.append(i)

# Chamando a funcao categorica
var_categoricas()

# Eliminar a variavel categorica Status
del list_var_categoricas[-1]

# Onehot da lib features engine
onehot = OneHotEncoder(variables = list_var_categoricas)
onehot.fit(df('Status', axis=1)[features])

# Transformando o dado original
df_fit = onehot.transform(df('Status', axis=1)[features])


# Seed para reproduzir o mesmo resultado
seed = 100
# Cria o balanceador SMOTE
balanceador = SMOTE(random_state = seed)

# Aplicar o balanceador
X_bal, y_bal = balanceador.fit_resample(df_fit, df["Status"])

# Treinando o modelo de machine learning

clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(X_bal, y_bal)

# Salvando o algoritmo
model = pd.Series(
    {
        "model": clf_tree,
        "onehot": onehot,
        "features": features,
        "target": y_bal
    }
)

model.to_pickle("../models/aprov_credito_tree_onehot.pkl")

model