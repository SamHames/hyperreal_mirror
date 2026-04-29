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
# # Worked Example: The Twenty Newsgroups
#
# The Twenty Newsgroups dataset is a 'classic' machine learning dataset, and an early
# example of a widely shared large scale text dataset.
#
# This notebook:
#
# - Downloads the source data
# - Reorganises how it is stored to make it more accessible for poking around and computational analysis.
# - Creates a HyperrealCorpus specific for this dataset, including handling all of the
#   details of working through newsgroup messages (including headers/metadata)
# - Creates a customised display format specifically for newsgroups messages
# - Uses the Hyperreal toolkit to create a clustering of words occuring in the body of
#   messages, and uses the web UI to serve this interface.

# %% [markdown]
# # Preparation I - Downloading the Data
#
# Let's start by downloading the data, if we haven't already.

# %%
from pathlib import Path
from urllib.request import urlretrieve

from IPython.display import HTML

# Where the data is coming from
data_url = "http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz"
# Where we're going to put it.
data_path = Path("data", "twenty_newsgroups")
tar_loc = data_path / "20news-19997.tar.gz"

# Download the data, but only if it isn't already present
if tar_loc.is_file():
    print("Data already downloaded.")
else:
    data_path.mkdir(exist_ok=True)
    print("Downloading original data file.")
    temp_location = tar_loc.with_suffix(".temp")
    urlretrieve(data_url, temp_location)
    temp_location.rename(tar_loc)


# %% [markdown]
# # Preparation II - Transform to a More Convenient Zip File
#
# The original format for this data is a [gzipped](https://en.wikipedia.org/wiki/Gzip)
# [tar file](https://www.loc.gov/preservation/digital/formats/fdd/fdd000531.shtml). This is a bit
# inefficient to use directly, especially since we want to access single
# files at a time. We'll therefore convert it to a zip file.
#
# We could also have extracted the files from the archive, but even with fast hardware
# dealing with lots of small files is usually slower than dealing with a few large files
# (plus the zip is much easier to download and poke at than 20,000 individual files).
#
# Note that we're not trying to preserve anything other than the structure and the file
# content - that is, there might be other file information present that we're going
# to ignore and just focus on the content of the files themselves.

# %%
import tarfile
import zipfile

data_loc = data_path / "20news-19997.zip"
temp_loc = data_loc.with_suffix(".temp")

if data_loc.is_file():
    print("Zip file already prepared")

else:
    print("Preparing Zip file")

    with tarfile.open(tar_loc, mode="r|gz") as tar_file:
        with zipfile.ZipFile(
            temp_loc,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as zip_data:
            # For every every member of the source file
            for member_info in tar_file:
                # Check if it's a file
                if member_info.isfile():
                    # And if it is a file, extract it from the tar
                    # and write it to the same location in the zip.
                    contents = tar_file.extractfile(member_info).read()
                    path = member_info.name
                    zip_data.writestr(path, contents)

    temp_loc.rename(data_loc)

display(
    HTML(
        '<a href="data/twenty_newsgroups/20news-19997.zip" download="20news-19997.zip">Download the prepared zip file</a>'
    )
)

# %% [markdown]
# # Overview of the Dataset
#
# This dataset is organised as twenty folders, each containing approximately 1000 files.
# If you look at one of the files in a text editor (as this has no file extension, you
# may need to right click and open with notepad or similar) you will see that it
# contains two parts:
#
# 1. A set of header lines that contain information about this message, in the format
#    header-name: header-content. This includes things like the `Subject:` line of the
#    message and the `Date:` it was sent.
# 2. After a blank line, there is the body of the message. There is no other
#    specification for how this is organised.
#
# This data format is heavily based on email. The relevant standard (at
# the time these materials were created) was
# [RFC 1036 - Standard for Interchange of USENET Messages](https://www.rfc-editor.org/rfc/rfc1036)

# %%
# Let's take a quick look at one of the files in it's entirety.
with zipfile.ZipFile(data_loc, "r") as newsgroups_zip:
    example_post_file = "20_newsgroups/misc.forsale/74150"
    # Read the contents of the file from the zip
    example_post_content = newsgroups_zip.read(example_post_file)
    # Display the contents. Note that we need to choose an older text encoding for these
    # files.
    print(example_post_content.decode("latin1"))

# %% [markdown]
# # Working Notes for the Following Sections
#
# Note that this implementation is a snapshot of the end of a long process and
# investigation. It didn't start like this: it was built incrementally as decisions
# were made about how to handle different complexities in the collected materials. The
# simplest decisions were about how to handle the various headers: the most difficult
# were how to contend with the body of posts as text-as-digital-media with a wide range
# of conventions for communication.
#
# While this is all a lot of decision making, it's very important to understand that
# this is not a complete or exact rendering: there are still aspects that haven't been
# considered here that are represented in the message: this doesn't include, for example
# the threaded nature of posts in reply to other posts.
#
# To make this whole thing more comprehensible, I'll break it down to smaller chunks
# with a more comprehensible flow, such as:
#
# 1. Interpreting the structure of a newsgroup post.
# 3. Deciding how to render newsgroup messages for reading in the web UI.
# 2. Working with the text of a message: handling conventionalised communication
#    structures like quoting, signatures and more. This will also include tokenisation.
# 4. Putting this together as a HyperrealCorpus
#    - What to index and how to index it
#    - What to display and how
#    - Accessing components of the messages efficiently on demand
#    - A more instructional overview of the core interface and why it's like that.

# %% [markdown]
# # Describing the Components of the Twenty Newsgroups
#
# Link to the standard kinds of preprocessing for twenty newsgroups, and why we don't
# want to use them.
#
# Making sense of the data format?
# - 1 folder per newsgroups, one file per message
# - Messages have a header, including things like subject and sender.
# - Messages have a body, the actual body of the text.
# - Things we're interested in:
#   - who's posting/where from?
#   - which newsgroups
#   - The text of the subject
#   - The text of the message itself.
#
# Need to show examples of what the data looks like, so we can decide what to do with it
# Possibly start with a very minimal example, just looking at the body text and one
# other piece of information, then look at having a more expansive example later?
#
# Also the option to troubleshoot and problem solve as we go, expanding our original
# definition.
#
# Note - as this notebook is aimed to giving you a guide to be read from end-to-end, we
# do some things that might make it harder for read for other purposes. For example,
# typically imports are grouped together at the beginning of the file, while we've
# introduced them one by one. Also we've duplicated several versions of this corpus,
# showing the simple starting point, and progressing to a more refined version -
# typically you'd just modify your existing definition instead of keeping all versions
# together [signpost version control].
#
# This section really needs to be broken down:
#
# - Text processing and decision making
# - Handling all the exceptions with the text documents
# - Rendering messages nicely for display/reading in the web context.
#


# %%
import collections
import functools
import re
from datetime import date
from email.utils import parsedate
from time import mktime

from tinyhtml import h, raw

from hyperreal import corpus, field_types


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

    matches = boundary_regex.finditer(text)

    split_start, last_end = 0, 0

    token_starts = []

    for match in matches:
        start, end = match.span()

        # If we've skipped over some content to get to the next split, it's time to
        # release a token, but including the boundaries.
        if start != last_end:
            token_starts.append(last_end)
            split_start = end

        last_end = end

    if token_starts and token_starts[-1] != text_length:
        token_starts.append(text_length)

    return [text[start:end] for start, end in zip(token_starts, token_starts[1:])]


example_string = " The cat, he sat on Matt's\nlap, really!\n"
print("Example string: ", example_string)
print("Tokenised for indexing: ", tokenise(example_string))
print("Tokenised for display: ", display_tokenise(example_string))


class TwentyNewsgroups(corpus.HyperrealCorpus):
    # Link to the documentation for this!
    def __init__(self):
        self.data_loc = data_loc
        self.signatures = self.get_signatures()

    def open_zip(self):
        return zipfile.ZipFile(self.data_loc, "r")

    def _all_post_paths(self):
        """
        Returns all files, including those with duplicate Message-ID headers.

        See https://www.rfc-editor.org/rfc/rfc1036 and the more recent
        https://www.rfc-editor.org/rfc/rfc5536#section-3.1.3 for details of Message-ID.

        """
        with self.open_zip() as z:
            for name in sorted(z.namelist()):
                yield name

    def _parse_post(self, raw_post):
        full_post = raw_post.decode("latin1")

        header, _, body = full_post.partition("\n\n")

        doc = {}

        for line in header.splitlines():
            header_key, _, value = line.partition(": ")
            doc[header_key] = value

        # Note that for a couple of reasons the dates here can be weird - so
        # we're ignoring any (potentially made up) timezones and just focusing
        # on the naive date of the timestamp.
        doc["Date"] = date.fromtimestamp(mktime(parsedate(doc["Date"])))
        doc["body"] = body

        return doc

    def all_doc_keys(self):
        """
        Returns all files after deduplicating based on the Message-ID header.

        Note that the Message-ID could be used as the key directly, but isn't directly
        mappable back to the original data file so the file path is used instead.

        """

        # Keep track of the messages we've seen already.
        seen_messages = set()

        with self.open_zip() as z:
            for path in self._all_post_paths():
                full_post_raw = z.read(path)
                post = self._parse_post(full_post_raw)

                # If we've already seen this Message-ID, skip onto the next.
                if post["Message-ID"] not in seen_messages:
                    yield path
                    seen_messages.add(post["Message-ID"])

    def docs(self, doc_keys):
        with self.open_zip() as z:
            for doc_key in doc_keys:
                full_post_raw = z.read(doc_key)
                yield doc_key, self._parse_post(full_post_raw)

    _QUOTE_RE = re.compile(
        "|".join(
            (
                r"writes in",
                r"writes:\s*$",
                r"wrote:\s*$",
                r"says:\s*$",
                r"said:\s*$",
                r"^\s*In article",
                r"^\s*Quoted from",
                r"^\s*\|",
                r"^\s*>",
                r"^\s*}",
                r"^\s*{",
                r"^\s*#",
                r"^\s*]",
                r"^\s*:",
                r"^\s*\+",
            )
        )
    )

    _PARA_DELIM = re.compile(r"\n\s*\n|^\s*--\s*$", re.MULTILINE)
    """
    Delimit paragraphs, or common signature block element starts (two or more hyphens as the whole line)
    """

    _STRIP_SIG_WHITESPACE = re.compile(r"\s+")

    def is_ignored_line(self, line):
        return self._QUOTE_RE.search(line)

    def get_signatures(self):
        """
        Find repeated signature blocks for removing from tokenisation.

        The last paragraph or block of text starting with --\\n will be
        treated as a potential signature, but it will only be removed if
        it's a duplicate of another signature block.

        This rule is intended to not remove blocks of text that are unique,
        as might happen if there's no clear demarcation between the end of
        the post and their signature/signoff (such as no paragraph break).

        """

        signature_counts = collections.Counter()

        for key, doc in self.docs(self.all_doc_keys()):
            paras = self._PARA_DELIM.split(doc["body"].strip())

            if len(paras) > 1:
                signature_counts[self._STRIP_SIG_WHITESPACE.sub("", paras[-1])] += 1

        return set(sig for sig, count in signature_counts.items() if count > 1)

    def mark_lines_ignore(self, body):
        lines = []

        paras = self._PARA_DELIM.split(body.strip())
        n_paras = len(paras)

        start = 0
        signature = ""
        end = n_paras

        # If there's no paragraph break, treat this as a message with no signature.
        if n_paras > 1:
            # Otherwise check if the last paragraph is in our signature blocks.
            if self._STRIP_SIG_WHITESPACE.sub("", paras[-1]) in self.signatures:
                signature = paras[-1]
                end -= 1

        for para in paras[start:end]:
            para += "\n\n"
            for line in para.splitlines(keepends=True):
                lines.append((self.is_ignored_line(line), line))

        if signature:
            for line in signature.splitlines(keepends=True):
                lines.append((True, line))

        return lines

    def doc_to_features(self, doc):
        indexed = {
            "subject": field_types.ValueSequence(tokenise(doc["Subject"])),
            "newsgroup": field_types.ValueSet(
                ng.strip() for ng in doc["Newsgroups"].split(",") if ng.strip()
            ),
            "from": field_types.Value(doc["From"].strip()),
            "date": field_types.RangeEncodableValue(doc["Date"]),
            # For validating that quoting behaviours are correctly handled.
            "line_start_character": field_types.ValueSet(
                line[0] for line in doc["body"].splitlines() if line
            ),
        }

        mark_lines = self.mark_lines_ignore(doc["body"])

        # The body text, handling quoting indicators at the start of lines.
        indexed["body"] = field_types.ValueSequence(
            [t for ignore, line in mark_lines if not ignore for t in tokenise(line)]
        )

        # Quoted text (inverse selection from the body)
        indexed["ignore"] = field_types.ValueSequence(
            t for ignore, line in mark_lines if ignore for t in tokenise(line)
        )

        if doc.get("Distribution", None):
            indexed["distribution"] = field_types.Value(doc["Distribution"].strip())
        if doc.get("Organization", None):
            indexed["organization"] = field_types.Value(doc["Organization"].strip())

        return indexed

    def doc_to_display_features(self, doc):
        indexed = {
            "subject": field_types.ValueSequence(display_tokenise(doc["Subject"])),
        }

        mark_lines = self.mark_lines_ignore(doc["body"])

        # The body text, handling quoting indicators at the start of lines.
        indexed["body"] = field_types.ValueSequence(
            t
            for ignore, line in mark_lines
            if not ignore
            for t in display_tokenise(line)
        )

        return indexed

    extra_css = """
        .search-hit details summary * {
            display: inline;
        }

        .search-hit :is(dt, dd) {
            display: inline;
        }

        .search-hit dt {
            font-weight: bolder;
        }

        .search-hit dt:after {
            content: ": ";
        }

        .search-hit {
            white-space: pre-wrap;
        }
    """

    def doc_to_html(self, doc, highlight_features=None):
        mark_lines = self.mark_lines_ignore(doc["body"])
        # render quoted text distinctly from other text.
        doc_body = [h("em")(line) if ignore else line for ignore, line in mark_lines]

        subject = doc.get("Subject", "<no subject>")

        display_fields = set(("From", "Newsgroups", "Date", "Organisation"))

        return h("details")(
            h("summary")(h("dl")(h("dt")("Subject"), h("dd")(subject))),
            h("dl", klass="stack")(
                *[
                    h("div")((h("dt")(key), h("dd")(doc[key])))
                    for key in display_fields
                    if key in doc
                ],
                (h("div")(h("dt")("body"), h("dd")(h("pre")(doc_body)))),
            ),
        )


# Actually initialise this corpus
newsgroups_corpus = TwentyNewsgroups()

# %% [markdown]
# # Using a Corpus
#
# Note that defining a corpus doesn't cause anything to happen by itself - it's like
# we've written a set of instructions, but haven't yet asked that these
# instructions be run. So let's actually create a concrete [instance]() of our
# corpus, and use that to do some of the basic things with our corpus.

# %%
from IPython.display import display_html

keys = list(newsgroups_corpus.all_doc_keys())[:5]

for key, doc in newsgroups_corpus.docs(keys):
    display_html(newsgroups_corpus.doc_to_html(doc))


# %% [markdown]
# # Creating an Index
#
# You might have created a corpus, and thought, so what? And you'd be more or less
# correct - the corpus describes the specifics of your document collection, but doesn't
# really let you do much with it. Most of the time, instead of using the corpus
# directly, you'll want to create a corpus and pass it to a HyperrealIndex to actually
# do something more interesting with it.
#

# %%
from datetime import date

from loky import get_reusable_executor

from hyperreal.index_core import HyperrealIndex
from hyperreal import query

pool = get_reusable_executor()

index_path = data_path / "twenty_newsgroups.db"

newsgroups_idx = HyperrealIndex(index_path, newsgroups_corpus, pool)

if not newsgroups_idx.indexed_field_summary:
    newsgroups_idx.rebuild(max_workers=10, doc_batch_size=2000, passage_size=16)


# %% [markdown]
# # Clustering Features for Search and Navigation
#

# %%
import random
import time

clustering = newsgroups_idx.plugins["feature_clusters"]

# clustering.delete_clusters(clustering.cluster_ids)

if not clustering.cluster_ids:
    # Set the state of the RNG to a consistent point.
    newsgroups_idx.random_state = random.Random(42)

    start_time = time.monotonic()

    # Initialise with a random clustering
    random_clustering = clustering.initialise_random_clustering(
        256, min_docs=10, include_fields=["body"]
    )

    clustering.replace_clusters(random_clustering)

    # Refine the clustering for a small number of iterations - we could go for longer,
    # but it usually doesn't matter as you'll spend the same amount of time examining
    # the output either way.
    clustering.refine_clustering(iterations=100)

    print(f"Clustering took: {time.monotonic() - start_time:.2f}")


# %%
import asyncio

from hyperreal.index_core import TableFilter
from hyperreal.web_server import serve_index


print("launching web server")

newsgroups_idx.facets = [
    (
        "Newsgroups",
        newsgroups_idx.field_features("newsgroup", min_docs=1),
        TableFilter(order_by="hits", first_k=20, keep_above=0),
    ),
    (
        "Organisations",
        newsgroups_idx.field_features("organization"),
        TableFilter(order_by="hits", first_k=20, keep_above=0),
    ),
    (
        "Posters",
        newsgroups_idx.field_features("from"),
        TableFilter(order_by="hits", first_k=20, keep_above=0),
    ),
]

newsgroups_idx.search_fields = {"body": tokenise, "subject": tokenise}

try:
    import os

    jupyter_hub_service_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/")
    url_base = jupyter_hub_service_prefix + "proxy/absolute/9999"

    loop = asyncio.get_running_loop()
    task = loop.create_task(serve_index(newsgroups_idx, base_path=url_base))

    display_link = h("a", href=url_base + "/browse/")("Browse Twenty Newsgroups")
    display(display_link)

except RuntimeError:
    loop = asyncio.new_event_loop()
    task = loop.create_task(serve_index(newsgroups_idx))
    loop.run_until_complete(task)


# %% [markdown]
# # Tabulation: a table of closest clusters and features for each newsgroup.
#
#

# %%
result_path = Path("results")
result_path.mkdir(exist_ok=True)

newsgroup_categories = newsgroups_idx.field_features("newsgroup", top_k_features=20)
all_newsgroups = newsgroups_idx.field_features("newsgroup")
top_clusters = TableFilter(order_by="jaccard_similarity", first_k=3)
top_features = TableFilter(order_by="jaccard_similarity", first_k=10)
top_cross_posted = TableFilter(order_by="hits", first_k=4)

header = h("thead")(
    h("tr")(
        h("th")(""),
        h("th", colspan=4)("Most similar clusters and features"),
    ),
    h("tr")(
        h("th")("Newsgroup"),
        h("th")("Top Cross-Posted Groups"),
        h("th")("Rank 1"),
        h("th")("Rank 2"),
        h("th")("Rank 3"),
    ),
)
table_rows = []
for newsgroup in sorted(newsgroup_categories):
    newsgroup_docs = newsgroups_idx[newsgroup][0]

    newsgroup_similarity = top_clusters(
        clustering.facet_clusters_by_query(newsgroup_docs)
    )

    similar_features = clustering.facet_clustering_by_query(
        newsgroup_docs, cluster_ids=newsgroup_similarity
    )

    row = []
    row.append(h("th")(newsgroup[1]))

    cross_posted = top_cross_posted(
        newsgroups_idx.facet_features(newsgroup_docs, all_newsgroups)
    )

    display_cross_posts = " ".join(f[1] for f in list(cross_posted)[1:]) + " ..."
    row.append(h("td")(display_cross_posts))

    for rank, cluster_id in enumerate(newsgroup_similarity):

        feature_stats = top_features(similar_features[cluster_id])
        display_features = " ".join(f[1] for f in feature_stats) + " ..."
        row.append(h("td")(display_features))

    table_rows.append(h("tr")(row))

visualisation_table = h("table")(header, h("tbody")(table_rows)).render()

with open(result_path / "twenty_newsgroups_top_clusters.html", "w") as table_out:
    table_out.write(visualisation_table)


# %% [markdown]
# # Tabulation: comparison keywords for each newsgroups.
#

# %%
import heapq
import math

keywords = collections.defaultdict(lambda: [(-1, ("", ""))] * 45)

N = newsgroups_idx.total_doc_count

for feature in newsgroups_idx.field_features("body", min_docs=10):
    feature_docs, doc_count, _ = newsgroups_idx[feature]

    for newsgroup in sorted(newsgroup_categories):
        news_docs, news_count, _ = newsgroups_idx[newsgroup]

        # chi-squared comparison, for now
        A = feature_docs.intersection_cardinality(news_docs)
        B = doc_count - A
        C = news_count - A
        D = N - (A + B + C)

        numer = (A * D - B * C) ** 2 * (A + B + C + D)
        denom = (A + B) * (C + D) * (B + D) * (A + C)
        chi_sq = numer / denom

        heapq.heappushpop(keywords[newsgroup], (chi_sq, feature))

header = h("thead")(
    h("tr")(
        h("th")("Newsgroup"),
        h("th")("Top Ranked Keywords (Chi-Squared)"),
    )
)
table_rows = []

for newsgroup, words in keywords.items():
    display_features = " ".join(f[1][1] for f in sorted(words, reverse=True))

    row = h("tr")(h("th")(newsgroup[1]), h("td")(display_features))

    table_rows.append(row)

visualisation_table = h("table")(header, h("tbody")(table_rows)).render()

with open(result_path / "twenty_newsgroups_top_keywords.html", "w") as table_out:
    table_out.write(visualisation_table)
