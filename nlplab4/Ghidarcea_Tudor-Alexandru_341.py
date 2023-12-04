import wikipedia
import gensim
import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import string
import numpy as np

# 1: am ales pagina wikipedia despre Anglia

page_title = "England"
wikipedia.set_lang("en")
page = wikipedia.page(page_title, auto_suggest=False)

# 2: incarcam modelele word2vec de la google si text8

modelword2vec = api.load('word2vec-google-news-300')
modeltext8=api.load('text8')

# 3: numaram cuvintele acoperite si neacoperite de modelul google

text = page.content
text = text.translate(str.maketrans('', '', string.punctuation)) #scoatem punctuatia
cuvinte = text.split()
cuvinte_distincte = set(cuvinte)
cuvinte_acoperite = set([word for word in cuvinte_distincte if word in modelword2vec.key_to_index])
cuvinte_neacoperite = cuvinte_distincte - cuvinte_acoperite
print(f"Cuvinte acoperite de modelword2vec: {len(cuvinte_acoperite)}")
print(f"Cuvinte neacoperite de modelword2vec: {len(cuvinte_neacoperite)}")

# 4: calculam similaritatea dintre perechi de cuvinte, le sortam dupa similaritate si afisam primele si ultimele 3 perechi
# dureaza mult


perechi = set()
for i, a in enumerate(cuvinte):
    if a not in modelword2vec.key_to_index.keys():
        continue
    for j, b in enumerate(cuvinte):
        if i >= j or b not in modelword2vec.key_to_index.keys():
            continue
        if a == b:
            continue
        perechi.add(tuple(sorted((a, b))))

similare = []
for pair in perechi:
    score = modelword2vec.similarity(pair[0], pair[1])
    similare.append((pair, score))

scoruri_sortate = sorted(set(similare), key=lambda x: x[1], reverse=True)

print("Cele mai similare 3 perechi: ")
for pair, score in scoruri_sortate[:3]:
    print(f"{pair[0]}, {pair[1]}: {score}")

print("Cele mai diferite 3 perechi: ")
for pair, score in scoruri_sortate[-3:]:
    print(f"{pair[0]}, {pair[1]}: {score}")

# 5: cele mai similare cuvinte pentru smart, king si big din ambele modele
# nu functioneaza pentru text8

cuvinte_de_comparat = ['smart', 'king', 'big']
for word in cuvinte_de_comparat:
    print(f"Word: {word}")
    print(f"Cuvinte similare google: {modelword2vec.most_similar(word)}")
    #print(f"Cuvinte similare text8: {modeltext8.most_similar(word)}")

# 6: reducem embedding-urile pentru cuvintele (...) la 2 dimensiuni si plotam cate un grafic pentru fiecare modelword2vec
cuvinte = ['car', 'motorcycle', 'bike', 'man', 'person', 'woman', 'child', 'king', 'queen', 'prince', 'plant', 'tree', 'flower']

# Google modelword2vec
embeddinguri_google = [modelword2vec[word] for word in cuvinte]
pca = PCA(n_components=2)
google_2d = pca.fit_transform(embeddinguri_google)
plt.scatter(google_2d[:, 0], google_2d[:, 1])

cuvinte=list(cuvinte)
for i, word in enumerate(cuvinte):
    plt.annotate(word, xy=(google_2d[i, 0], google_2d[i, 1]))

plt.title("Embedding-uri pentru Word2vec:")
plt.show()

embeddinguri_text8 = []

for word in cuvinte:
    if word in modeltext8:
        embeddinguri_text8.append(modeltext8[word])
        embeddinguri_text8 = np.array(embeddinguri_text8)
pca = PCA(n_components=2)


# nu merge pentru text8
#text8_2d = pca.fit_transform(embeddinguri_text8)
#plt.scatter(text8_2d[:, 0], text8_2d[:, 1])

#for i, word in enumerate(cuvinte):
#    plt.annotate(word, xy=(text8_2d[i, 0], text8_2d[i, 1]))

plt.title("Embedding-uri pentru text8: ")
plt.show()