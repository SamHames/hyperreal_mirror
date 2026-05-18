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
# # Explore and Search Your Transcripts
#
# This notebook takes a [zip file](https://en.wikipedia.org/wiki/ZIP_(file_format)) full of transcripts in Word document format (`.docx`) and makes them searchable and explorable.
#
# ## Requirements and assumptions
#
# You will need to put all your transcript files into a zip file - see [documentation for Windows](https://support.microsoft.com/en-us/windows/zip-and-unzip-files-8d28fa72-f2f9-712f-67df-f80cf89fd4e5) and [documentation for Mac](https://support.apple.com/en-au/guide/mac-help/mchlp2528/mac). The folders and filenames for where the transcript is stored inside the zip file will be *preserved and displayed*. If your filenames are meaningful you will get more out of this tool.
#
# Files other than Word documents will be ignored - but note that it will take longer to upload if you include other files.
#
# This search tools assumes you use the common convention of creating your transcripts in the form:
#
# ```
# speaker_code: what they said
# ```
#
# Where speaker_code is the name or identifier of the speaker, the colon (`:`) and tab characters separate the speaker code from the transcribed text of what they said.


# %% [markdown]
# ## Upload your transcripts
#
# Run the cell of code below (`# RUN ME`) to create a button to upload your zip file.
#
# You can run the cell by placing your cursor inside it by clicking on the text of the cell, then clicking the play button ("▶") in the top menu bar.
#
# When the button appears click on it to select your zip file.

# %%
# RUN ME
from ipywidgets import FileUpload, Layout, Output, Button, VBox
from process_transcripts import run_process_from_jupyter

upload_zip = FileUpload(
    accept=".zip",
    description="click here to upload your transcripts",
    layout=Layout(width="50%", height="80px"),
)
display_output = Output()

run_button = Button(description="Process transcripts")
run_button.on_click(run_process_from_jupyter(upload_zip, display_output))

display(VBox([upload_zip, run_button, display_output]))
