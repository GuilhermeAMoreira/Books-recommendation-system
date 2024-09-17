import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# Inicialização do ratings
ratings = pd.read_csv('data/ratings_normalized.csv', delimiter=';', low_memory=False)

# Criar uma tabela de usuário x livros com ratings
user_book_ratings = ratings.pivot(index='UserID', columns='ISBN', values='Rating').fillna(0)

# Converter para matriz esparsa
X = csr_matrix(user_book_ratings.values)

# Aplicar SVD para encontrar a matriz de fatoração
svd = TruncatedSVD(n_components=50, random_state=42)
factor_matrix = svd.fit_transform(X)

# Função para recomendar livros com base em filtragem colaborativa
def recommend_books(user_index, factor_matrix, user_book_ratings, num_recommendations=5):
    user_ratings = factor_matrix[user_index]
    similarity = user_ratings.dot(factor_matrix.T)
    similar_users = similarity.argsort()[::-1][1:]
    
    # Obter livros mais bem avaliados pelos usuários semelhantes
    recommendations = []
    for similar_user in similar_users:
        recommended_books = user_book_ratings.iloc[similar_user].sort_values(ascending=False).index
        recommendations.extend(recommended_books)
        if len(recommendations) >= num_recommendations:
            break
    return recommendations[:num_recommendations]

# Exemplo de recomendação para o usuário 1
print(recommend_books(user_index=0, factor_matrix=factor_matrix, user_book_ratings=user_book_ratings))
