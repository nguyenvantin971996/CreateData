import pandas as pd

df_homes = pd.read_csv("data_18_18_yen.csv")
df_homes1 = pd.read_csv("data_18_18_yen_2.csv")
df_homes2 = pd.read_csv("data_18_18_yen_3.csv")

pd.concat([df_homes, df_homes1, df_homes2]).to_csv('data_18_yen_2.csv', index=False)