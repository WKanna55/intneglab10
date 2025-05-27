import pandas as pd

df = pd.read_excel('Data10.xlsx')

df.to_csv('Datacsvisado.csv', index=False)