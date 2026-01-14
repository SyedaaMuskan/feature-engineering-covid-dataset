from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
import pandas as pd
from sklearn.impute import SimpleImputer
df=pd.read_csv("covid_toy.csv")
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df.drop(columns=["has_covid"]),df["has_covid"],test_size=0.2)
from sklearn.compose import ColumnTransformer
t=ColumnTransformer(transformers=[("tnf1",SimpleImputer(),["fever"]),('tnf2',OneHotEncoder(),['gender','city']),
                       ('tnf3',OrdinalEncoder(categories=[['Mild','Strong']]),['cough'])],remainder='passthrough')
p=t.fit_transform(x_train)
d=t.transform(x_test)
print(p)
print(d)