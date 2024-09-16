import pandas as pd

# Carregar o dataset
books = pd.read_csv('data/Books.csv', delimiter = ';', low_memory=False)
ratings = pd.read_csv('data/Ratings.csv', delimiter = ';', low_memory=False)
users = pd.read_csv('data/Users.csv', delimiter = ';', low_memory=False)


# Remover linhas com valores nulos em colunas críticas
books = books.dropna(subset=['Title', 'Author'])
ratings = ratings.dropna(subset=['UserID', 'ISBN'])
users = users.dropna(subset=['UserID'])

ratings['UserID'] = ratings['UserID'].astype(str)

# Remover livros duplicados
books = books.drop_duplicates(subset=['ISBN'])
# Remover avaliações duplicadas (se um usuário avaliou o mesmo livro várias vezes)
ratings = ratings.drop_duplicates(subset=['UserID', 'ISBN'])
# Filtrar avaliações com valor zero ou extremamente baixos, caso não sejam úteis
ratings = ratings[ratings['Rating'] > 0]

# Contar o número de avaliações por usuário
user_counts = ratings['UserID'].value_counts()
# Filtrar usuários que avaliaram menos de 5 livros, por exemplo
ratings = ratings[ratings['UserID'].isin(user_counts[user_counts >= 5].index)]

# Contar o número de avaliações por livro
book_counts = ratings['ISBN'].value_counts()
# Filtrar livros que receberam menos de 5 avaliações, por exemplo
ratings = ratings[ratings['ISBN'].isin(book_counts[book_counts >= 5].index)]

user_mean = ratings.groupby('UserID')['Rating'].mean()

# Calcular a média das avaliações por usuário
user_mean = ratings.groupby('UserID')['Rating'].mean().reset_index()

# Renomear a coluna para diferenciar
user_mean.rename(columns={'Rating': 'User-Mean'}, inplace=True)
# Fazer o merge das médias de volta no dataframe original
ratings = ratings.merge(user_mean, on='UserID', how='left')
# Normalizar as avaliações
ratings['Rating-Normalized'] = ratings['Rating'] - ratings['User-Mean']
# Verificar o resultado
print(ratings[['UserID', 'Rating', 'User-Mean', 'Rating-Normalized']].head())
# Salvar a matriz normalizada em um arquivo CSV
ratings.to_csv('data/ratings_normalized.csv', index=False)

ratings = ratings.reset_index()


# Mostrar as primeiras linhas do dataset
print(books.head())
print(ratings.head())
print(users.head())

import matplotlib.pyplot as plt

# Plotar a distribuição das avaliações normalizadas
ratings['Rating-Normalized'].hist(bins=20)
plt.title('Distribuição das Avaliações Normalizadas')
plt.xlabel('Avaliação Normalizada')
plt.ylabel('Frequência')
plt.show()

print(ratings['Rating-Normalized'].describe())
