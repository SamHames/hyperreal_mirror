# %% [markdown]
# # Working with conversation data


# %% [markdown]
# # Requirements and assumptions


# %%
# Create a folder to hold the conversations
from pathlib import Path

conversations_path = Path("conversations")
conversations_path.mkdir(exist_ok=True)

# %% [markdown]
# # Upload your transcripts
#
# You can either click the button below, or upload your files directly in the
# conversations folder from the lefthand panel.

# %%
import ipywidgets

uploaded_via_jupyter = False

uploader = ipywidgets.FileUpload(accept=".docx", multiple=True)
display(uploader)


# %% [markdown]
# Process the uploaded documents.
#
# Uploaded files are saved in the conversations folder. All .docx files in that folder
# will be included in the processing below.

# %%
from pathlib import Path

for uploaded_file in uploader.value:
    with open(conversations_path / uploaded_file.name, "wb") as f:
        f.write(uploaded_file.content)


# %% [markdown]
# # Pre-process conversations
#
# This step will aim to extract each turn of each uploaded conversation, and separate
# the speaker from the turn content. This step will only be as consistent as your
# transcripts are.
#
# This will completely delete and recreate the state of processed conversations - if
# you want to upload new or edited files, make the contents of the conversations folder
# match what you want, and re-run this cell.

# %%
# Start by setting up a small database to hold the processed information.

import sqlite3

convo_db_path = conversations_path / "conversations.db"

convo_db = sqlite3.connect(convo_db_path, isolation_level=None)

convo_db.executescript(
    """
    DROP table if exists turn;
    CREATE table turn (
        turn_id integer primary key,
        source_file,
        turn_no,
        speaker,
        turn,
        unique(source_file, turn_no)
    )

    """
)

# Then we'll identify and load conversation turns from each transcript.
import glob

from docx import Document

convo_db.execute("begin")

for transcript_filepath in glob.glob(str(conversations_path / "**.docx")):
    filename = Path(transcript_filepath).relative_to(conversations_path)
    print("Processing", transcript_filepath)

    with open(transcript_filepath, "rb") as transcript_file:

        # Load the word document
        doc = Document(transcript_file)

        # We're going to treat each paragraph in the word file as a turn
        for turn_no, paragraph in enumerate(doc.paragraphs):
            para_text = paragraph.text

            # Identify speakers by looking for the colon character then a tab.
            # If colon-tab is not matched, no speaker will be assigned to this turn.
            speaker_split = para_text.split(":\t")

            if len(speaker_split) == 2:
                speaker, text = speaker_split
            else:
                speaker, text = None, para_text

            convo_db.execute(
                "INSERT into turn values (?, ?, ?, ?, ?)",
                (None, transcript_filepath, turn_no, speaker, text),
            )

convo_db.execute("commit")

# %% [markdown]
#
# The previous step created a little database holding the extracted turns from all
# uploaded files. Now we're going to index this dataset in a way that is aware of the
# within and across-turn structure of conversations.


# %% [markdown]
#
# First lets define two functions for breaking down turns into word-like units
# (tokenisation). We'll create two tokenisers - the first one normalises case by
# lowercasing all of the text, then finds and splits the turn up at characters
# indicating word boundaries('\b'), or whitespace characters like space, newlines, and
# tabs.
#
# The second, display_tokenise, breaks on the same places, but does not lowercase, or
# remove spaces, so we can recreate and highlight search result matches/concordances
# of search terms.

# %%

import re  # Python's inbuilt regular expression library

boundary_regex = re.compile(r"(\b|\s+)")


def tokenise(text):
    """
    Break down text of posts into words.

    This is a simple tokeniser that breaks on whitespace and simple word 'boundaries'
    based on character classes (for example, at punctuation).

    """
    return [
        stripped
        for token in boundary_regex.split(text.lower())
        if (stripped := token.strip())
    ]


def display_tokenise(text):
    """Tokenise, but preserving punctuation and whitespace for display as original."""
    text_length = len(text)

    # Find split matches, padding with a split at the start/end - if it's not needed
    # they will be merged with existing matches.
    matches = [
        (0, 0),
        *[m.span() for m in boundary_regex.finditer(text)],
        (text_length, text_length),
    ]
    merged = [matches[0]]

    for start, end in matches[1:]:

        last_start, last_end = merged[-1]

        if start == last_end:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))

    token_starts = [m[1] for m in merged]

    return [text[start:end] for start, end in zip(token_starts, token_starts[1:])]


# %% [markdown]
# # Index the transcripts
#
# Now we're going to make the transcripts searchable using words by indexing with the
# the hyperreal library. This involves constructing a corpus that describes what we want
# to consider a 'document' for the purposes of retrieval, display and search.
#
# This indexing process is aware of basic conversational structure however: instead of
# indexing just the text of the individual turn, we break things down by considering:
#
# 1. Whether a token occurs in position initial, final, or medial.
# 2. Whether a token occurs in the first, second, or third turn. This means that each
#    turn is indexed three times, in the three relative positions, to make it easier to
#    construct queries that are relative to this structure.

# %%
from tinyhtml import h

from hyperreal import field_types
from hyperreal.corpus import HyperrealCorpus


class TranscriptCorpus(HyperrealCorpus):

    @property
    def db(self):
        if not hasattr(self, "_db"):
            self._db = sqlite3.connect(convo_db_path, isolation_level=None)
        return self._db

    def __getstate__(self):
        return

    def __setstate__(self, *args):
        self._db = None

    def all_doc_keys(self):
        return (r[0] for r in self.db.execute("select turn_id from turn"))

    def docs(self, doc_keys):

        for key in doc_keys:
            # key points to the current turn as the relative turn 1 -> we also retrieve
            # the following two turns as relative turns 2 and 3.
            turns = self.db.execute(
                """
                WITH rolling_turn as (
                    select 
                        source_file, 
                        turn_no
                    from turn
                    where turn_id = ?
                )
                SELECT 
                    turn_id,
                    source_file,
                    turn_no,
                    turn_no - (select turn_no from rolling_turn) + 1 as turn_offset,
                    coalesce(speaker, '<no speaker>') as speaker,
                    turn
                from turn
                where source_file = (select source_file from rolling_turn)
                    and turn_no 
                        between (select turn_no from rolling_turn) and 
                            (select turn_no + 2 from rolling_turn) 
                """,
                [key],
            )

            names = [col[0] for col in turns.description]

            turn_rows = []

            for turn in turns:
                turn_data = {k: v for k, v in zip(names, turn)}

                source_file = turn_data["source_file"]
                parts = Path(source_file).parts

                turn_data["file"] = parts[-1]
                turn_data["path"] = "/".join(parts[1:-1])

                turn_rows.append(turn_data)

            yield key, turn_rows

    def doc_to_features(self, doc):
        indexed = {}

        for key in ("turn_no", "file", "path", "speaker"):
            indexed[key] = field_types.Value(doc[0][key])

        for turn_data in doc:

            turn_offset = turn_data["turn_offset"]
            turn_field = f"t{turn_offset}"

            turn_tokens = tokenise(turn_data["turn"])

            if turn_tokens:
                pos_initial = turn_tokens[0]
                pos_final = turn_tokens[-1]
                pos_medial = turn_tokens[1:-1]

                indexed[turn_field + ",initial"] = field_types.Value(pos_initial)
                indexed[turn_field + ",final"] = field_types.Value(pos_final)
                indexed[turn_field + ",medial"] = field_types.ValueSequence(pos_medial)
                indexed[turn_field] = field_types.ValueSequence(turn_tokens)

            indexed[turn_field + ",speaker"] = field_types.Value(turn_data["speaker"])

        return indexed

    def doc_to_display_features(self, doc):

        # Note we need to be careful here because turn 2/3 will not exist at the last
        # of the transcript
        display = {}

        for i, turn in enumerate(doc):
            display[f"t{i+1}"] = display_tokenise(doc[i]["turn"])

        return display

    def features_to_html_concordance(
        self, doc_features, display_features, highlight_features
    ):
        """Do not render concordances: turns are short already."""
        return None

    def render_turn(self, turn_no, speaker, turn):
        if speaker:
            return (turn_no, " ", h("em")(speaker), ":\t", turn)
        else:
            return (turn_no, turn)

    def doc_to_html(self, doc, highlight_features=None):

        turn_block = self.db.execute(
            """
            SELECT turn_no, speaker, turn, turn_id - ?1 + 1 as relative_turn
            from turn
            where turn_id between ?1 - 2 and ?1 + 2
            order by turn_id
            """,
            [doc[0]["turn_id"]],
        )

        return h("div")(
            h("h3")(doc[0]["path"], " ", doc[0]["file"]),
            h("ol")(
                h("li", klass=f"turn{turn[-1]}")(self.render_turn(*turn[:3]))
                for turn in turn_block
            ),
        )

    extra_css = """
        .turn1 {
            background-color: lightgray;
        }
    """

    def close(self):
        if hasattr(self, "_db"):
            self._db.close()


# %% [markdown]
#
# The above TranscriptCorpus only describes how to work with this collection of
# documents - it doesn't actually do anything yet. Let's make it actually do
# something.
#

# %%
from loky import get_reusable_executor

from hyperreal.index_core import HyperrealIndex, TableFilter

pool = get_reusable_executor()

convo_corpus = TranscriptCorpus()

convo_idx = HyperrealIndex(
    conversations_path / "conversations_index.db", convo_corpus, pool
)

convo_idx.rebuild()

# %%
clustering = convo_idx.plugins["feature_clusters"]

include_fields = ["t1", "t2", "t3"]

random_clustering = clustering.initialise_random_clustering(
    include_fields=include_fields, min_docs=3, n_clusters=64
)
clustering.replace_clusters(random_clustering)
refined = clustering.refine_clustering(iterations=50)

# %%
import asyncio
from hyperreal.web_server import serve_index


print("launching web server")

search_fields = include_fields + ["speaker"]

for i in range(1, 4):
    for subfield in ("initial", "final", "medial"):
        search_fields.append(f"t{i},{subfield}")

convo_idx.facets = [
    (
        "Speaker",
        convo_idx.field_features("speaker", min_docs=1),
        TableFilter(order_by="hits", first_k=20, keep_above=0),
    ),
    (
        "File",
        convo_idx.field_features("file"),
        TableFilter(order_by="hits", first_k=20, keep_above=0),
    ),
    (
        "Path",
        convo_idx.field_features("path"),
        TableFilter(order_by="hits", first_k=20, keep_above=0),
    ),
]

convo_idx.search_fields = {field: tokenise for field in search_fields}

try:
    import os

    jupyter_hub_service_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/")
    url_base = jupyter_hub_service_prefix + "proxy/absolute/9999"

    loop = asyncio.get_running_loop()
    task = loop.create_task(serve_index(convo_idx, base_path=url_base))

    display(h("a", href=url_base + "/browse/")("Browse your conversations here"))

except RuntimeError:
    loop = asyncio.new_event_loop()
    task = loop.create_task(serve_index(convo_idx))
    loop.run_until_complete(task)

# %%
