# About #

# About

Hyperreal is a Python tool for interactive qualitative analysis of large
collections of documents. Hyperreal builds on ideas from topic modelling, corpus linguistics and full text search. The aim is to enable new kinds of qualitative analysis workflows for amounts of data that would otherwise be too big to approach using existing approaches. It is designed to work with very large collections with only modest resources: interactive performance is always a development priority. 

This library allows you to take a large collection of documents such as the 55 million questions and answers from [Stack Overflow](https://stackoverflow.com) and:

1. Create a searchable index of the content across many different fields, allowing you pay attention to the different ways a complex platform can be used.
2. Use a text analytics algorithm to create an exploratory structure (simple clusters of features) for understanding this large dataset in context.
3. Use the software and web interface to interactively query and display matching documents and parts of documents for each of the individual feature clusters.
4. Edit the resulting clusters of features to begin addressing your research question by aligning your reading of the content with the query used to retrieve it.

Hyperreal aims to enables fine grained control over how documents are indexed, and, unlike other libraries, enables fine-grained control over how they are *displayed* in different contexts. Reading documents in context is a core concern.

See [A Recent History of Python Through Stack Overflow Questions and Answers](https://www.youtube.com/watch?v=scpjoqtgrtE) for an overview of how this can be used in practise.

## Quickstart

### Installation

Hyperreal requires the installation of [the Python programming language](https://www.python.org/downloads/).

Hyperreal can be installed using Pip from the command line (
[Windows](https://learn.openwaterfoundation.org/owf-learn-windows-shell/introduction/introduction/#windows-command-shell),
[Mac](https://support.apple.com/en-au/guide/terminal/apd5265185d-f365-44cb-8b09-71a064a42125/mac))
by running the following commands:

```
python -m pip install hyperreal
```

### Usage

Hyperreal can be used in three different ways to flexibly support different
use cases:

- as a command line application
- as a Python library
- as a local web application

All of hyperreal's functionality is available from the Python library, but you
will need to write Python code to use it directly. The command line interface
allows for quick and repeatable experimentation and automation for standard
data types - for example if you often work with StackExchange data the command line
will allow you to rapidly work with many different StackExchange data collections.
The web application is currently focused solely on creating and interactive
editing of models built on top of existing indexes.


### Command Line

The following script gives a basic example of using the command line interface
for hyperreal. This will work for cases where you have a plain text file
(here called `corpus.txt`), with each `document` in the collection on its own
line.

If you haven't worked with the command line before, you might find the
following resources useful:

- [Software Carpentry resources for Mac](https://swcarpentry.github.io/shell-novice/)
- [Open Water Foundation resources for Windows](https://learn.openwaterfoundation.org/owf-learn-windows-shell/)

```
# Create a corpus database from a plaintext file in the current working directory
hyperreal plaintext-corpus create corpus.txt corpus.db

# Create an index from the corpus
hyperreal plaintext-corpus index corpus.db corpus_index.db

# Create a model from that index, in this case with 128 clusters and
# only include features present in 10 or more documents.
hyperreal model corpus_index.db --min-docs 10 --clusters 128

# Use the web interface to serve the results of that modelling
# After running this command point your web browser to http://localhost:8080
hyperreal plaintext-corpus serve corpus.db corpus_index.db

```

### Library

This example script performs the same steps as the command line example.

```python

from hyperreal import corpus, index

# create and populate the corpus with some documents
c = corpus.PlainTextSqliteCorpus('corpus.db')

with open('corpus.txt', 'r') as f:
  # This will drop any line that has no text (such as a paragraph break)
  docs = (line for line in f if line.strip())
  c.replace_docs(docs)


# Index that corpus - note that we need to pass the corpus object for
# initialisation.
idx = index.Index('corpus_index.db', corpus=c)
# This only needs to be done once, unless the corpus changes.
idx.rebuild()

# Create a model on this index, with 128 clusters and only including features
# that match at least 10 documents.
idx.initialise_clusters(n_clusters=128, min_docs=10)
# Refine the model for 10 iterations. Note that you can continue to refine
# the model without initialising the clusters.
idx.refine_clusters(iterations=10)

# Inspect the output of the model using the index instance (currently quite
# limited). This will print the top 10 most frequent features in each
# cluster.
for cluster_id in idx.cluster_ids:
    cluster_features = idx.cluster_features(cluster_id)
    for feature in cluster_features[:10]:
        print(feature)

# Perform a boolean query on the corpus, looking for documents that contain
# both apples AND oranges in the text field.
q = i[('text', 'apples')] & i[('text', 'oranges')]
# Lookup all of the documents in the corpus that match this query.
docs = idx.get_docs(q)

# 'Pivot' the features in the index with respect to all cluster in the model.
#  This will show the top 10 features in each cluster that are similar to the
#  query.
for cluster_detail in idx.pivot_clusters_by_query(query, top_k=10):
    print(cluster_detail)

# This will show the top 10 features for a selected set of cluster_ids.
for cluster_detail in idx.pivot_clusters_by_query(query, cluster_ids=[3,5,7], top_k=10):
    print(cluster_detail)

```
