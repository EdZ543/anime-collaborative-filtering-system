"""The main application file for the Gradio app."""

import os
import gradio as gr
import pandas as pd
import requests
import torch
import torch.nn as nn

MAL_CLIENT_ID = os.getenv("MAL_CLIENT_ID")

anime_indexes = pd.read_csv("./data/anime_indexes.csv")
animes = anime_indexes["Anime"].values.tolist()

anime_embeddings = pd.read_csv("./data/anime_embeddings.csv", header=None)
anime_embeddings = torch.tensor(anime_embeddings.values)


def fetch_anime_image_url(anime_id):
    url = f"https://api.myanimelist.net/v2/anime/{anime_id}?fields=main_picture"
    headers = {"X-MAL-CLIENT-ID": MAL_CLIENT_ID}
    response = requests.get(url, headers=headers)
    image_url = response["main_picture"]["large"]
    return image_url


def recommend(anime):
    anime_index = anime_indexes[anime_indexes["Anime"] == anime].index[0]
    anime_embedding = anime_embeddings[anime_index][None]

    embedding_distances = nn.CosineSimilarity(dim=1)(anime_embeddings, anime_embedding)
    recommendation_indexes = embedding_distances.argsort(descending=True)[1:7].tolist()
    recommendations = [
        (
            "https://cdn.myanimelist.net/images/anime/1600/134703.jpg",
            anime_indexes.iloc[index]["Anime"],
        )
        for index in recommendation_indexes
    ]

    return recommendations


with gr.Blocks() as space:
    gr.Markdown(
        """
    # Anime Collaborative Filtering System
    This is a Pytorch recommendation model that uses neural collaborative filtering.
    Enter an anime, and it will suggest similar shows!
    """
    )

    dropdown = gr.Dropdown(label="Enter an anime", choices=animes)

    gallery = gr.Gallery(label="Recommendations", rows=2, columns=3)

    dropdown.change(fn=recommend, inputs=dropdown, outputs=gallery)

space.launch()
