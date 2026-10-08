"""querychat + Groq demo: LLM-powered data filtering on the Titanic dataset.

Run with: shiny run app.py

Ask it: "Show only women who survived" or "filter to first class passengers"
"""

from pathlib import Path

import querychat
from chatlas import ChatGroq
from dotenv import load_dotenv
from seaborn import load_dataset
from shiny import App, render, ui

# Load .env from the same directory as this script, regardless of CWD
load_dotenv(Path(__file__).parent / ".env")

# data -----
titanic = load_dataset("titanic")

# querychat setup -----
qc = querychat.QueryChat(
    titanic,
    "titanic",
    client=ChatGroq(model="qwen/qwen3.8-27b"),
)

# ui -----
app_ui = ui.page_sidebar(
    qc.sidebar(),
    ui.card(
        ui.card_header(ui.output_text("title")),
        ui.output_data_frame("data_table"),
        fill=True,
    ),
    fillable=True,
    title="Titanic Explorer (Groq)",
)


# server -----
def server(input, output, session):
    qc_vals = qc.server()

    @render.text
    def title():
        return qc_vals.title() or "Titanic dataset"

    @render.data_frame
    def data_table():
        return qc_vals.df()


app = App(app_ui, server)
