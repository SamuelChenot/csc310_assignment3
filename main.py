import pandas as pd
from sklearn import tree
from treeviz import tree_print
from sklearn.metrics import accuracy_score

df = pd.read_csv("tennis.csv")
df.head()

features_df = df.drop(['play'], axis = 1)
features_df.head()

target_df = pd.DataFrame(df['play'])
target_df.head()

dtree = tree.DecisionTreeClassifier(criterion='entropy')

dtree.fit(features_df,target_df)

