import pandas as pd

data=pd.read_csv("tennis.csv")
features=data.columns[:-1]
target=data.columns[-1]
classes=data[data.columns[-1]].unique()
overall_impurity=1
for cl in classes:
    n = len(data[data[target]==cl])
    overall_impurity-=(n/len(data[target]))**2
print(overall_impurity)

def lowestImpurity(data,features):
    branch_impurity={}
    best_feature=["None",1]
    for feature in features:
        feature_impurity=0
        impurities={}
        branches = data[feature].unique()
        for branch in branches:
            gi=1
            branch_data=data[data[feature]==branch]
            counts=branch_data[target].value_counts()
            for cl in classes:
                n=counts[cl] if cl in counts else 0
                gi-=(n/len(branch_data))**2
            feature_impurity+=(len(branch_data)/len(data))*gi
            impurities.update({branch:float(gi)})
        if feature_impurity<best_feature[1]:
            best_feature[0]=feature
            best_feature[1]=float(feature_impurity)
            branch_impurity=impurities
    return best_feature[0],branch_impurity



def makeTree(data,features,target):
    if len(data[target].unique())==1:
        return data[target].unique()[0]
    if len(features)==0:
        return data[target].mode()[0]
    root,branches=lowestImpurity(data,features)
    tree={root:{}}
    new_features=features.drop(root)
    for branch in branches:
        subdata=data[data[root]==branch]
        if subdata.empty:
            tree[root][branch]=data[target].mode()[0]
        else:
            subtree=makeTree(subdata,new_features,target)
            tree[root][branch]=subtree
    return tree

decisionTree=makeTree(data,features,target)
print(decisionTree)
