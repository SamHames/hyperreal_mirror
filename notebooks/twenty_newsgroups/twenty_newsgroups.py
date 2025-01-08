# %% [markdown]
# # Example: Twenty Newsgroups
#
# The twenty newsgroups dataset is a 'classic' dataset, and one of the first examples of
# a large scale text dataset. This example walks through the process of creating a
# corpus for a particular collection of documents from scratch, including outline some
# of the necessary data transformation steps we need to take a long the way. Note that
# this notebook illustrates the messier steps in the iterative process of understanding
# and modelling a document collection. If you just want to see a complete example, you
# can [skip to the complete notebook]().

# %% [markdown]
# # Preparation I - downloading the data
#
# Let's start by downloading the data, if we haven't already.

# %%
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


# %% [markdown]
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
            for member_info in tar_file:
                if member_info.isfile():
                    contents = tar_file.extractfile(member_info).read()
                    path = member_info.name
                    zip_data.writestr(path, contents)

    temp_loc.rename(data_loc)


# %% [markdown]
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

# %%
import re

from hyperreal import corpus, index_builder

boundary_regex = re.compile(r"\b")


def tokeniser(text):
    return [token.strip() for token in boundary_regex.split(text) if token.strip()]


class TwentyNewsgroups1(corpus.Corpus):

    # Link to the documentation for this!
    def __init__(self):
        self.data_loc = data_loc

    def open_zip(self):
        return zipfile.ZipFile(self.data_loc, "r")

    def all_doc_keys(self):
        with self.open_zip() as z:
            for name in z.namelist():
                yield name

    def docs(self, doc_keys):
        with self.open_zip() as z:
            for key in doc_keys:
                yield key, z.read(key).decode("utf8")

    def indexable_docs(self, doc_keys):
        for key, doc in self.docs(doc_keys):
            yield key, {"text": tokeniser(doc)}


# %% [markdown]
# # Using the Corpus
#
# Note that defining a corpus doesn't cause anything to happen by itself, we've
# described how things should work, but we haven't actually done anything with it yet.
# So let's use our corpus to do something. We'll create an `Index` for our corpus, and
# use that to examine the implications of how we've chosen to represent the documents.

# %%
corp = TwentyNewsgroups1()
keys = list(corp.all_doc_keys())[:10]
for _, doc in corp.indexable_docs(keys):
    print(doc)
