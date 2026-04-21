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
                (h("div")(h("dt")("body"), h("dd")(doc_body))),
            ),
        )


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
    newsgroups_idx.rebuild(max_workers=10, doc_batch_size=2000, passage_size=16)

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


# +
# Visualisation: a table of closest clusters and features for each newsgroup.
#
#

from tinyhtml import h
from hyperreal.index_core import TableFilter

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


# +
# Comparison table: keywords for each newsgroups.
#
#

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


# +
import asyncio
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
    task = loop.create_task(serve_index(convo_idx, base_path=url_base))

    display(h("a", href=url_base + "/browse/")("Browse Twenty Newsgroups"))

except RuntimeError:
    loop = asyncio.new_event_loop()
    task = loop.create_task(serve_index(newsgroups_idx))
    loop.run_until_complete(task)
