"""The main application file for the Gradio app."""

import os
import gradio as gr
import pandas as pd
import requests
import torch
import torch.nn as nn
from urllib import parse

MAL_CLIENT_ID = os.getenv("MAL_CLIENT_ID")

anime_indexes = pd.read_csv("./data/anime_indexes.csv")
animes = anime_indexes["Anime"].values.tolist()

anime_embeddings = pd.read_csv("./data/anime_embeddings.csv", header=None)
anime_embeddings = torch.tensor(anime_embeddings.values)


def fetch_anime_image(anime):
    query_url = f"https://api.myanimelist.net/v2/anime?q={parse.quote(anime)}&limit=1"
    headers = {"X-MAL-CLIENT-ID": MAL_CLIENT_ID}
    query_response = requests.get(query_url, headers=headers)

    try:
        image_url = query_response.json()["data"][0]["node"]["main_picture"]["large"]
        return image_url
    except:
        return None


def recommend(anime):
    anime_index = anime_indexes[anime_indexes["Anime"] == anime].index[0]
    anime_embedding = anime_embeddings[anime_index][None]

    embedding_distances = nn.CosineSimilarity(dim=1)(anime_embeddings, anime_embedding)
    recommendation_indexes = embedding_distances.argsort(descending=True)[1:].tolist()

    recommendations = []
    for recommendation_index in recommendation_indexes:
        recommendation_anime = anime_indexes.iloc[recommendation_index]["Anime"]
        recommendation_image = fetch_anime_image(recommendation_anime)
        if recommendation_image is not None:
            recommendations.append((recommendation_image, recommendation_anime))

        if len(recommendations) == 5:
            break

    return recommendations


css = """
#selection_column {align-items: center}
"""

with gr.Blocks(css=css) as space:
    gr.Markdown(
        """
    # Anime Collaborative Filtering Recommender
    This is a Pytorch recommendation model that uses neural collaborative filtering.
    Enter an anime, and it will suggest similar shows!
    """
    )

    with gr.Box():
        gr.Markdown("Enter an anime:")

        with gr.Column(elem_id="selection_column"):
            dropdown = gr.Dropdown(container=False, choices=animes)
            selection_image = gr.Image(show_label=False, width=225, visible=False)

    gallery = gr.Gallery(label="Recommendations", columns=[2, 2, 3, 3, 4, 5])

    def submit(anime):
        if anime is None:
            return {
                selection_image: gr.update(visible=False),
                gallery: gr.update(value=[]),
            }

        selection_image_url = fetch_anime_image(anime)
        recommendations = recommend(anime)

        return {
            selection_image: gr.update(visible=True, value=selection_image_url),
            gallery: gr.update(value=recommendations),
        }

    dropdown.change(fn=submit, inputs=dropdown, outputs=[selection_image, gallery])

space.launch()
