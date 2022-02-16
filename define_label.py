import pandas as pd

df = pd.read_csv("data/twitter_clean_all.csv", encoding = 'utf-8')

df_count = df['y'].value_counts(sort = True)
print(df_count.head(10))

labels = ["{smile}", "{solid-sad}", "{dunno}", "{hope}", "{mad}"]
df_sort = df[df['y'].isin(labels)]

print("sorted length: ", len(df_sort))

df_count = df_sort['y'].value_counts(sort = True)

print(df_count)

df_sort.to_csv('data/twitter_clean.csv', index=False)