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
# # Explore and Search A Table of Text Data
#
# This notebook takes as input a tabular (or tabular-like) file, and creates a
# browsable, searchable index view for close reading across the entries in the table.
#
# ## Supported Files Formats
#
# - An Excel spreadsheet (`.xlsx`) with tabular structure.
#


# %% [markdown]
# ## Run this notebook and upload your tabular or tabular-like data.
#
# Run this notebook by hitting the `▶▶` button in the menu above - you'll be asked if
# you want to restart this notebook - answer 'yes' and the notebook will be run.
#
# Upload your tabular-like data with the "Upload your data" button.
# Follow the prompts to make choices about what to include in your analysis.
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
from tabular_ui import UI

ui = UI()

# %%
