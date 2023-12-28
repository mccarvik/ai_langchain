
import sys
sys.path.append("..")
from config import set_environment
set_environment()

from langchain.embeddings.openai import OpenAIEmbeddings
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi



def sample_embeddings():
    embeddings = OpenAIEmbeddings()
    text = "This is a sample query."
    query_result = embeddings.embed_query(text)
    print(query_result)
    print(len(query_result))

def sample_documents():
    words = ["cat", "dog", "computer", "animal"]
    embeddings = OpenAIEmbeddings()
    doc_vectors = embeddings.embed_documents(words)
    print(doc_vectors)
    X = np.array(doc_vectors)
    dists = squareform(pdist(X))
    df = pd.DataFrame(
        data=dists,
        index=words,
        columns=words
    )
    plt.figure(figsize=(10, 6))
    styled_df = df.style.background_gradient(cmap='coolwarm')
    dfi.export(styled_df, 'embed_heatmaps.png')
    # # Save the styled DataFrame as an image
    # fig, ax = plt.subplots(figsize=(6, 4))  # Set the figure size
    # ax.axis('off')  # Turn off axis for better visualization
    # styled_df.set_table(ax)
    # plt.savefig('embed_heatmaps.png', bbox_inches='tight')
    # plt.close()


# sample_embeddings()
sample_documents()  