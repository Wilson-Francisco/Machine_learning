# importando as biblitecas
import pandas as pd
from sklearn import pipeline
from sklearn import tree
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
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


var_categoricas()

# Onehor da lib features engine
onehot = OneHotEncoder(variables = list_var_categoricas)
target = "Status"

# Nossos objeto de modelo
clf_tree = tree.DecisionTreeClassifier(max_depth=5, random_state = 42)

# Seed para reproduzir o mesmo resultado

# Normalizador  as variaveis
norm = MinMaxScaler()

# Nosso pipeline com todos objetos
model_pipeline = pipeline.Pipeline(steps = [("onehot", onehot),
                                            ("norm ", norm ),
                                            ("Tree", clf_tree)] )

# Divisão das variaveis de Treino e Teste.
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target] , test_size = 0.3, random_state = 42)

# Ajustando o modelo
model_pipeline.fit(X_train, y_train)

# Salvando o algoritmo
model = pd.Series(
    {
        "model": model_pipeline,
        "features": features,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "target": y_test
    }
)
model.to_pickle("../models/aprov_credito_tree_pipeline.pkl")

model

print(model_pipeline)