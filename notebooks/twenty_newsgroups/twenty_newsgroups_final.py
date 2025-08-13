# # Example: Twenty Newsgroups
#
# The twenty newsgroups dataset is a 'classic' dataset, and one of the first examples of
# a large scale text dataset. This example takes one version of the twenty newsgroups
# dataset and creates an explorable version of the dataset. Note that this is the final
# version outline in [the other pedagogical one].

# # Preparation I - downloading the data
#
# Let's start by downloading the data, if we haven't already.

# +
from pathlib import Path
from urllib.request import urlretrieve

# Where the data is coming from
data_url = "http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz"
# Where we're going to put it.
data_path = Path("data")
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
# -


# # Preparation II - transforming the data to a more convenient container.
#
# The original format for this data is a [gzipped]() [tar file](). This is a bit
# inconvenient to use directly, both in general and for our purposes, so we'll first
# convert it to a [zip file](). We could also have extracted the files from the
# archive, but even with fast hardware dealing with lots of small files is usually a
# worst case scenario for performance (plus the zip is much easier to download and poke
# at than 20,000 individual files).
#
# Note that we're not trying to preserve anything other than the structure and the file
# content - that is, there is plenty of other metadata present that we're going to
# ignore.

# +
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
            for member_info in tar_file:
                if member_info.isfile():
                    contents = tar_file.extractfile(member_info).read()
                    path = member_info.name
                    zip_data.writestr(path, contents)

    temp_loc.rename(data_loc)
# -


# # Making a Corpus for Twenty Newsgroups: First Pass
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

# +
import re
import collections
from datetime import date
from email.utils import parsedate
from time import mktime

from tinyhtml import h, raw

from hyperreal import corpus


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


class TwentyNewsgroups(corpus.HyperrealCorpus):

    extra_css = """
        .snippets {
            --space: var(--s-2);
            margin: var(--s-2);
        }
        .snippets li {
            list-style-type: roman;
            margin-inline-start: var(--s2);
        }

        .doc dt::after {
            content: ":";
        }

        .doc dl {
            margin: var(--s0);
            --space: var(--s-2);
        }
    """

    # Link to the documentation for this!
    def __init__(self):
        self.data_loc = data_loc

        self.repeated_blocks = self.identify_repeated_blocks()

    para_splitter = re.compile(r"\n\n|^-+\s*$", re.MULTILINE)

    def identify_repeated_blocks(self):

        para_counter = collections.Counter()

        for key, doc in self.docs(self.all_doc_keys()):

            for para in self.para_splitter.split(doc["body"]):
                if stripped := para.strip():
                    para_counter[stripped] += 1

        return {para for para, count in para_counter.items() if count > 1}

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

    def _split_signature(self, body):
        """
        Based on the scikit-learn heuristic:

        https://github.com/scikit-learn/scikit-learn/blob/f0ab589f/sklearn/datasets/twenty_newsgroups.py#L124

        """

        lines = body.strip().split("\n")
        n_lines = len(lines)

        # Note that we depart from scikit-learn, and don't check the final content line
        # We already know this can't be non-whitespace because of strip, and if it is
        # all '-' characters, we want to continue further into the message.
        line_num = 0

        for line_num in range(n_lines - 2, -1, -1):
            line = lines[line_num]
            if line.strip().strip("-") == "":
                break

        if line_num > 0:
            return (
                "\n".join(lines[:line_num]),
                "\n".join(lines[line_num:]),
                line_num,
                n_lines,
            )
        else:
            return body.strip(), "", n_lines, n_lines

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
                r"writes:$",
                r"wrote:$",
                r"says:$",
                r"said:$",
                r"^In article",
                r"^Quoted from",
                r"^\|",
                r"^>",
                r"^}",
                r"^{",
                r"^#",
                r"^]",
                r"^:",
                r"^\+",
            )
        )
    )

    def is_quoted_line(self, line):
        return self._QUOTE_RE.search(line)

    def mark_lines_ignore(self, body):

        lines = []

        for block in self.para_splitter.split(body):
            block_lines = block.splitlines(keepends=True)
            if block_lines:
                block_lines[-1] += "\n\n"

            if block.strip() in self.repeated_blocks:
                for line in block_lines:
                    lines.append((True, line))

            else:
                for line in block_lines:
                    lines.append((self.is_quoted_line(line), line))

        return lines

    def indexable_docs(self, doc_keys):
        for doc_key, doc in self.docs(doc_keys):
            indexed = {
                "subject": tokenise(doc["Subject"]),
                "newsgroup": set(
                    ng.strip() for ng in doc["Newsgroups"].split(",") if ng.strip()
                ),
                "from": doc["From"],
                "date": doc["Date"],
                # For validating that quoting behaviours are correctly handled.
                "line_start_character": {
                    line[0] for line in doc["body"].splitlines() if line
                },
            }

            mark_lines = self.mark_lines_ignore(doc["body"])

            # The body text, handling quoting indicators at the start of lines.
            indexed["body"] = [
                t for ignore, line in mark_lines if not ignore for t in tokenise(line)
            ]

            # Quoted text (inverse selection from the body)
            indexed["ignore"] = [
                t for ignore, line in mark_lines if ignore for t in tokenise(line)
            ]

            if doc.get("Distribution", None):
                indexed["distribution"] = doc["Distribution"]
            if doc.get("Organization", None):
                indexed["organization"] = doc["Organization"]

            yield doc_key, indexed

    def _render_html_doc(self, doc, snippets=None, snippet_summary=None):

        mark_lines = self.mark_lines_ignore(doc["body"])
        # render quoted text distinctly from other text.
        doc["body"] = (h("em")(line) if ignore else line for ignore, line in mark_lines)

        subject = doc.pop("Subject")

        concordances = None
        if snippets:
            concordances = h("details")(
                h("summary")(h("em")(snippet_summary)),
                h("ol", klass="stack snippets")([h("li")(line) for line in snippets]),
            )

        return (
            h("details")(
                h("summary")(subject),
                h("dl", klass="stack")(
                    (h("dt")(key), h("dd")(h("pre")(value)))
                    for key, value in doc.items()
                ),
            ),
            concordances,
        )

    def html_docs(self, doc_keys, highlight_features=None):

        body_values = []
        if highlight_features is not None:
            body_values = {v for f, v in highlight_features if f == "body"}

        if body_values:
            for doc_key, doc in self.docs(doc_keys):
                tokenised = [
                    t
                    for ignore, line in self.mark_lines_ignore(doc["body"])
                    if not ignore
                    for t in tokenise(line)
                ]
                matches = [
                    i for i, value in enumerate(tokenised) if value in body_values
                ]

                matched_tokens = sorted(set(tokenised[m] for m in matches))
                summary_matches = f"{len(matches)} matches on: " + " ".join(
                    matched_tokens
                )

                # Add spaces back to tokens
                tokenised = [t + " " for t in tokenised]

                # highlight matches
                for m in matches:
                    tokenised[m] = h("mark")(tokenised[m])

                snippets = [
                    raw(h("span")([tokenised[max(0, i - 10) : i + 11]]).render())
                    for i in matches
                ]

                yield doc_key, self._render_html_doc(
                    doc, snippets=snippets, snippet_summary=summary_matches
                )

        else:
            for doc_key, doc in self.docs(doc_keys):
                yield doc_key, self._render_html_doc(doc)


newsgroups_corpus = TwentyNewsgroups()


# for line, count in line_counter.items():
#     if count > 2:
#         print(line, count)

# print(sum(line_counter.values()))
# print(sum(c for c in line_counter.values() if c > 2))
# assert False


# repeated lines:

# for doc, lines in newsgroups_corpus.match_all_lines("wrote:$"):
#     print(doc)
#     for line in lines:
#         print(line)
#     print()
# -

# # Using a Corpus
#
# Note that defining a corpus doesn't cause anything to happen by itself - it's like
# we've written a set of instructions, but haven't actually asked that these
# instructions be run yet. So let's actually create a concrete [instance]() of our
# corpus, and use that to do some of the basic things with our corpus.
#

# +
# from IPython.display import display_html

# newsgroups_corpus = TwentyNewsgroups()
# keys = list(newsgroups_corpus.all_doc_keys())[:1]

# for key, doc in newsgroups_corpus.html_docs(keys):
#     display_html(doc)

# -

# # Creating an Index
#
# You might have created a corpus, and thought, so what? And you'd be more or less
# correct - the corpus describes the specifics of your document collection, but doesn't
# really let you do much with it. Most of the time, instead of using the corpus
# directly, you'll want to create a corpus and pass it to an Index to actually do
# something more exciting.
#

# +
from datetime import date

from loky import get_reusable_executor

from hyperreal.index_core import HyperrealIndex
from hyperreal import query

pool = get_reusable_executor()

index_path = data_path / "twenty_newsgroups.db"

newsgroups_idx = HyperrealIndex(index_path, newsgroups_corpus, pool)

if not newsgroups_idx.indexed_field_summary:
    newsgroups_idx.rebuild(max_workers=10, doc_batch_size=2000, passage_group_size=32)

# print(
#     sum(
#         len(doc["body"])
#         for _, _, doc in newsgroups_idx.indexable_docs(newsgroups_idx.all_doc_ids())
#     )
# )


# ngs = newsgroups_idx.field_features("newsgroup")

# pivot = newsgroups_idx.facet_features(ngs, newsgroups_idx[("body", "sanctions")][0])

# renderer = FeatureStatsRenderer(
#     newsgroups_idx,
#     order_by_stat="doc_count",
#     display_stats=["jaccard_similarity"],
#     top_k=40,
# )

# renderer.to_rows(pivot)

##

# +


# assert False

# print(newsgroups_idx.total_doc_count, "doc_count")

# test_features = [
#     ("body", "banana"),
#     ("body", "ban", "ban\U0010ffff"),
#     ("date", date(1993, 5, 1)),
#     ("date", date(1993, 4, 1), date(1993, 4, 15)),
# ]

# for feature in test_features:
#     search_results = newsgroups_idx[feature]
#     print(feature, len(search_results[0]), search_results[1:])

# phrase = [("body", "to"), ("body", "be")]

# # print(len(newsgroups_idx.match_any(body, dates)))
# # print(len(newsgroups_idx.match_all(body, dates)))

# phrase_docs = newsgroups_idx.match_phrase(*phrase)

# print(len(phrase_docs))

# for _, _, doc in newsgroups_idx.docs(phrase_docs[:3]):
#     print(doc["body"])


# q = query.MatchAny(test_features)

# ser = list(q.to_index_rows(newsgroups_idx))
# print(ser)

# newsgroups_idx.defined_queries = {"match_any": query.MatchAny}

# deser = query.deserialize(newsgroups_idx, ser)
# print(deser)

# print(deser.evaluate(newsgroups_idx))

# assert False


# ## TODO: demonstrate querying and faceting framework.
# # -

# # # Create a Clustering of Features (Words)
# #
# # Now let's do something more interesting and create a `clustering` of the features in
# # this collection of documents.

# # +

# # feature_cluster is a default plugin on the index, we're just giving it a shorter name
# newsgroups_clusters = newsgroups_idx.p.feature_clusters

# # Select some features from the index that we'll use to make a clustering. We'll choose
# # from the 'subject' and the 'body'. Since subject and body lines of messages are both
# # discursively and functionally different we'll choose to create a clustering that
# # doesn't mix the two groups.


# # We're going to
# # -


# # # What can we do with a Clustering?
# #
import random
import time

clustering = newsgroups_idx.plugins["feature_clusters"]

# clustering.delete_clusters(clustering.cluster_ids)

if not clustering.cluster_ids:

    # Set the state of the RNG to a consistent point.
    clustering.random_state = random.Random(42)

    start_time = time.monotonic()

    random_clustering = clustering.initialise_random_clustering(
        256, min_docs=10, include_fields=["body"]
    )

    clustering.replace_clusters(random_clustering)

    clustering.refine_clustering(
        group_test_n_clusters=16,
        iterations=50,
        random_group_checks=1,
        use_passages=True,
    )

    print(f"Clustering took: {time.monotonic() - start_time:.2f}")

# +
import asyncio
from hyperreal.web_server import serve_index
from hyperreal.index_core import TableFilter

print("launching web server")

newsgroups_idx.facets = [
    (
        "The Twenty Newsgroups",
        newsgroups_idx.field_features("newsgroup", top_k_features=20),
        TableFilter(order_by="hits", first_k=20),
    ),
    (
        "Top Organisations",
        newsgroups_idx.field_features("organization"),
        TableFilter(order_by="hits", first_k=20, keep_above=0),
    ),
]

try:
    loop = asyncio.get_running_loop()
    task = loop.create_task(
        serve_index(newsgroups_idx, base_path="/proxy/absolute/9999")
    )

except RuntimeError:
    loop = asyncio.new_event_loop()
    task = loop.create_task(serve_index(newsgroups_idx))
    loop.run_until_complete(task)


# -

# # Launch a Server to Interactively Explore This Model
#
# Having seen the basic building blocks of what an index can do, let's explore this
# collection through the web interface.

# +
# query = newsgroups_idx.match_any(FieldValues("body", values=["gay"]))
# body_fields = newsgroups_idx.field_values("body", min_docs=10)
# faceted = newsgroups_idx.facet_count(body_fields, query)

# print(faceted.order_by("jaccard_similarity", keep_n_values=20))
