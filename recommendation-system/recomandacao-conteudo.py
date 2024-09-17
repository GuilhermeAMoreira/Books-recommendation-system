from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Usar a coluna 'Book-Title' para calcular a similaridade
tfidf = TfidfVectorizer(stop_words='english')

# Remover NaNs e aplicar o TF-IDF
books['Title'] = books['Title'].fillna('')
tfidf_matrix = tfidf.fit_transform(books['Title'])

# Calcular a similaridade entre os livros
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Função de recomendação baseada em conteúdo
def recommend_books_content(book_title, cosine_sim=cosine_sim):
    idx = books[books['Title'] == book_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    
    book_indices = [i[0] for i in sim_scores]
    return books['Title'].iloc[book_indices]

# Exemplo de recomendação para o livro "Harry Potter"
print(recommend_books_content('Harry Potter and the Philosopher\'s Stone'))
