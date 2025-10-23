import pandas as pd
from sklearn.tree import DecisionTreeClassifier


data=pd.read_csv("tennis.csv")
features=data.columns[:-1]
target=data.columns[-1]
tree=DecisionTreeClassifier()
tree.fit(data[features],data[target])
print(tree)