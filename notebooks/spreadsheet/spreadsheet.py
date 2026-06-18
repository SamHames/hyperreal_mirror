# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Explore and Search Text in a Spreadsheet
#
# This notebook takes a spreadsheet (in `.xlsx`) format and makes the text searchable
# and browsable using the feature clustering approach of Hyperreal.
#
# ## Minimum requirements
#
# 1. Your spreadsheet needs to have tabular (rectangular) structure. The first row of
# each sheet is treated as the header and the contents of those rows will be used as
# the column names for selection and display.
# 2. The text you want to analyse must be in one or more columns of the same table.
# 3. Fields you want to include for filtering/selection must be all of the same type,
# such as dates, strings, integers or floating point numbers.
# 4. For best results any contextual information about the text you want to analyse
# should be presented in
# [tidy format](https://tidyr.tidyverse.org/articles/tidy-data.html).
# If your data is in a tidy format, and the column names are consistent across tables
# relationships will be automatically inferred. This will make joinable contextual
# information ready for display and analysis.


# %% [markdown]
# ## Run this notebook and upload your spreadsheet
#
# Run this notebook by hitting the `▶▶` button in the menu above - you'll be asked if
# you want to restart this notebook - answer 'yes' and the notebook will be run.
#
# Upload your spreadsheet (`.xlsx`) file with the "Upload your spreadsheet" button.
# Follow the prompts to make choices about what to include in your analysis.
#
#
# Workflow notes:
# Choose the text table: chooses the granularity of documents
# Select one or more columns as the main documents
# Can infer relational properties from column names, or can have some extra syntax
# for linking together?
# Choose from relational properties and properties in the main table as the key.
#
# MVP: Choose a table and text columns.

# %%
from io import BytesIO

from ipywidgets import FileUpload, Layout, Output, Button, VBox
from spreadsheet_corpus import serve_spreadsheet

upload_spreadsheet = FileUpload(
    accept=".xlsx",
    description="Upload your spreadsheet",
    layout=Layout(width="90%", height="2lh"),
)
display_output = Output()

def run_process(clicked_button):
    with display_output:
        serve_spreadsheet(BytesIO(upload_spreadsheet.value[0].content))

run_button = Button(description="Explore spreadsheet")
run_button.on_click(run_process)

display(VBox([upload_spreadsheet, run_button, display_output]))
