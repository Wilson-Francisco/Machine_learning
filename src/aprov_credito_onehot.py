# Importando as biblitecas
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import tree
from sklearn.model_selection import train_test_split
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


# Separar as variaveis categoricas
list_var_categoricas = []
def var_categoricas():
    list_var_categoricas.clear()
    for i in df.columns[0:21].tolist():
        if df.dtypes[i] == 'object' or df.dtypes[i] == 'category':
            list_var_categoricas.append(i)

# Chamando a funcao categorica
var_categoricas()


# Onehot da lib features engine
onehot = OneHotEncoder(variables = list_var_categoricas)
onehot.fit(df.drop('Status', axis=1)[features])

# Transformando o dado original
df_fit = onehot.transform(df.drop('Status', axis=1)[features])


# Seed para reproduzir o mesmo resultado
seed = 100
# Cria o balanceador SMOTE
balanceador = SMOTE(random_state = seed)

# Aplicar o balanceador
X_bal, y_bal = balanceador.fit_resample(df_fit, df["Status"])

# Divisão das variaveis de Treino e Teste.
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size = 0.3, random_state = 42)

# Treinando o modelo de machine learning

clf_tree = tree.DecisionTreeClassifier(max_depth=5, random_state = 42)
clf_tree.fit(X_train, y_train)

# Salvando o algoritmo
model = pd.Series(
    {
        "model": clf_tree,
        "onehot": onehot,
        "features": features,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "target": y_test
    }
)

model.to_pickle("../models/aprov_credito_tree_onehot.pkl")

model