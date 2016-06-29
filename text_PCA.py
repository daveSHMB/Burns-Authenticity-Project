from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
from txt_input import initial_read
import os
from io import open


text_files = os.path.join(os.path.dirname(__file__), 'texts')
files = os.listdir(text_files)

X = []  # Stores data on each input text
y = []  # Stores authors of input texts
markers = []  # Stores only unique author names for plotting purposes

# Loop through each text file and append relevant data to lists
for doc in os.listdir(text_files):

    text = open(text_files + '\\' + doc, 'r', encoding='utf-8')
    # open the file, store as lower case
    text = text.read().lower()

    X.append(initial_read(text))

    author = doc.split()

    y.append(author[0].title())
    if author[0].title() not in markers:
        markers.append(author[0].title())



# Convert both X and Y to numpy arrays - not 100% sure why this is important
X = np.array(X)
y = np.array(y)

X_std = StandardScaler().fit_transform(X)
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)


with plt.style.context('fivethirtyeight'):
    plt.figure(figsize=(18, 7), dpi=96)

    # Amend this so a random but unique colour is selected for each author
    for lab, col in zip(markers, ('blue', 'red', 'black', 'green', 'orange', 'pink')):

        plt.scatter(Y_sklearn[y == lab, 0],
                    Y_sklearn[y == lab, 1],
                    label=lab,
                    c=col,
                    s=40)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='right')
    plt.tight_layout()
    plt.show()