"""
The backend of the transcript searching tool.

This primarily handles processing word documents in the uploaded zip file into a small
SQLite database, and defines the TranscriptCorpus object and some simple utility 
functions to create and launch the web interface for it.

"""

import asyncio
from io import BytesIO
from pathlib import Path
import os
import sqlite3
import re
from zipfile import ZipFile

from docx import Document
from loky import get_reusable_executor
from tinyhtml import h

from hyperreal import field_types
from hyperreal.corpus import HyperrealCorpus
from hyperreal.index_core import HyperrealIndex, TableFilter
from hyperreal.web_server import serve_index


def create_transcript_db_from_zip(transcript_db_path, open_zip):

    # Remove the existing db, if it exists at all
    transcript_db_path.unlink(missing_ok=True)

    transcript_db = sqlite3.connect(transcript_db_path, isolation_level=None)

    transcript_db.executescript(
        """
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

    transcript_db.execute("begin")

    transcript_count = 0
    with ZipFile(open_zip) as zipf:
        for name in zipf.namelist():
            # Only process docx files
            if not name.lower().endswith(".docx"):
                continue

            transcript_count += 1

            with zipf.open(name, "r") as transcript_file:
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

                    transcript_db.execute(
                        "INSERT into turn values (?, ?, ?, ?, ?)",
                        (None, name, turn_no, speaker, text),
                    )

    transcript_db.execute("commit")
    print(f"Processed {transcript_count} transcripts")


# Python's inbuilt regular expression library

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


class TranscriptCorpus(HyperrealCorpus):

    def __init__(self, transcript_db_path):
        self.transcript_db_path = transcript_db_path
        self._db = None

    @property
    def db(self):
        if self._db is None:
            self._db = sqlite3.connect(self.transcript_db_path, isolation_level=None)
        return self._db

    def __getstate__(self):
        return self.transcript_db_path

    def __setstate__(self, *args):
        self.transcript_db_path = args[0]
        self._db = None

    def close(self):
        if self._db is not None:
            self._db.close()
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
            return (turn_no, ":\t", turn)

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
            h("h3")(doc[0]["file"]),
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


def create_index(transcript_corpus, index_db_path):
    """Create an index for the given corpus"""

    pool = get_reusable_executor()

    transcript_idx = HyperrealIndex(index_db_path, transcript_corpus, pool)

    # Generate the index.
    transcript_idx.rebuild(doc_batch_size=5000)

    return transcript_idx


def run_clustering(transcript_idx):
    """Run the clustering for the corpus."""
    clustering = transcript_idx.plugins["feature_clusters"]

    include_fields = ["t1"]

    random_clustering = clustering.initialise_random_clustering(
        include_fields=include_fields, min_docs=3, n_clusters=64
    )
    clustering.replace_clusters(random_clustering)
    refined = clustering.refine_clustering(iterations=50)


def launch_web_server(transcript_idx):

    search_fields = ["t1", "speaker"]

    # TODO: work out how to skin this for conversations compared to transcripts.
    # for i in range(1, 4):
    #     for subfield in ("initial", "final", "medial"):
    #         search_fields.append(f"t{i},{subfield}")

    transcript_idx.facets = [
        (
            "Speaker",
            transcript_idx.field_features("speaker", min_docs=1),
            TableFilter(order_by="hits", first_k=20, keep_above=0),
        ),
        (
            "File",
            transcript_idx.field_features("file"),
            TableFilter(order_by="hits", first_k=20, keep_above=0),
        ),
        (
            "Path",
            transcript_idx.field_features("path"),
            TableFilter(order_by="hits", first_k=20, keep_above=0),
        ),
    ]

    transcript_idx.search_fields = {field: tokenise for field in search_fields}

    jupyter_hub_service_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/")
    url_base = jupyter_hub_service_prefix + "proxy/absolute/9999"

    loop = asyncio.get_running_loop()
    task = loop.create_task(serve_index(transcript_idx, base_path=url_base))

    return h("a", href=url_base + "/browse/")("Browse your transcripts")


class run_process_from_jupyter:

    def __init__(self, file_upload_widget, display_output):
        self.file_upload_widget = file_upload_widget
        self.display_output = display_output

    def __call__(self, button):

        with self.display_output:
            print("Extracting transcripts from:", self.file_upload_widget.value[0].name)
            uploaded_zip = BytesIO(self.file_upload_widget.value[0].content)

            transcript_db_path = Path("transcripts.db")
            index_db_path = Path("transcript_idx.db")

            create_transcript_db_from_zip(transcript_db_path, uploaded_zip)

            print("Indexing transcripts")
            transcript_corpus = TranscriptCorpus(transcript_db_path)
            transcript_idx = create_index(transcript_corpus, index_db_path)

            print("Running clustering algorithm")
            run_clustering(transcript_idx)

            display(launch_web_server(transcript_idx))
