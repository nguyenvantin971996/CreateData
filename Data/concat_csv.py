import pandas as pd

df_homes = pd.read_csv("data_10_node_yen.csv")
df_homes1 = pd.read_csv("data_10_node_yen_1.csv")
df_homes2 = pd.read_csv("data_10_node_yen_2.csv")

pd.concat([df_homes, df_homes1, df_homes2]).to_csv('data_10_node_yen.csv', index=False)