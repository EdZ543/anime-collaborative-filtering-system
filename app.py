import os
import gradio as gr
import pandas as pd
import numpy as np
import requests
from urllib import parse
from dotenv import load_dotenv

load_dotenv()

anime_indexes = pd.read_csv("./data/anime_indexes.csv")
animes = anime_indexes["Anime"].values.tolist()

MAL_CLIENT_ID = os.getenv("MAL_CLIENT_ID")


def fetch_anime_image(anime):
    query_url = f"https://api.myanimelist.net/v2/anime?q={parse.quote(anime)}&limit=1"
    headers = {"X-MAL-CLIENT-ID": MAL_CLIENT_ID}
    query_response = requests.get(query_url, headers=headers)
    image_url = query_response.json()["data"][0]["node"]["main_picture"]["large"]
    return image_url


def recommend(anime):
    return None


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

    gallery = gr.Gallery(label="Recommendations")

    def submit(anime):
        if anime is None:
            return {
                selection_image: gr.update(visible=False),
                gallery: gr.update(value=[]),
            }

        selection_image_url = fetch_anime_image(anime)

        return {
            selection_image: gr.update(visible=True, value=selection_image_url),
        }

    dropdown.change(fn=submit, inputs=dropdown, outputs=[selection_image, gallery])

space.launch()
