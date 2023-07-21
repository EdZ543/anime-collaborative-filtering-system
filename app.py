"""The main application file for the Gradio app."""

import gradio as gr
import pandas as pd
import torch

animes_df = pd.read_csv("./data/animes.csv")
anime_embeddings_df = pd.read_csv("./data/anime_embeddings.csv", header=None)

title_list = animes_df["Title"].tolist()
embeddings = torch.tensor(anime_embeddings_df.values)


def recommend(index):
    embedding = embeddings[index]

    embedding_distances = torch.nn.CosineSimilarity(dim=1)(embeddings, embedding)
    recommendation_indexes = embedding_distances.argsort(descending=True)[1:4]

    recommendations = []
    for rank, recommendation_index in enumerate(recommendation_indexes):
        recommendation = animes_df.iloc[int(recommendation_index)]
        value = recommendation["Image URL"]
        label = f'{rank + 1}. {recommendation["Title"]}'
        recommendations.append((value, label))

    return recommendations


css = """
.gradio-container {align-items: center}
#container {max-width: 795px}
"""


with gr.Blocks(css=css) as space:
    with gr.Column(elem_id="container"):
        gr.Markdown(
            """
        # Anime Collaborative Filtering System
        This is a Pytorch recommendation model that uses neural collaborative filtering.
        Enter an anime, and it will suggest similar shows!
        """
        )

        dropdown = gr.Dropdown(label="Enter an anime", choices=title_list, type="index")

        gallery = gr.Gallery(label="Recommendations", rows=1, columns=3, height="265")

        dropdown.change(fn=recommend, inputs=dropdown, outputs=gallery)

space.launch()
