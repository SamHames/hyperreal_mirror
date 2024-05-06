# Design of Hyperreal #

# About the Design

Hyperreal is a software tool for computational interpretive analysis of collections of documents. This document describes the design, including the goals, overall architecture, and workings of major components of the library.

# Aims and Goal

Hyperreal aims to enable flexible analysis of many kinds of documents in your qualitative research framework of choice, while working within a modest compute footprint. In particular Hyperreal aims to enable analysis of collections of material that would otherwise be too large for qualitative analysis.

The detailed aims of the Hyperreal project are to:

- Allow modelling of (mostly static) large collections of complex documents as more than just text, including structured metadata, image features and more.
- Allow modelling of documents at different levels of granularity, as both a whole and as parts.
- Allow starting simply by modelling documents as basic data types and incrementally adding customised behaviour only when necessary.
- Provide scaffolding for qualitative analysis of very large datasets with transparent and comprehensible algorithmic support.
- Enable researcher agency for interpretive decision making through appropriate affordances for *interactive* editing, transformation, recreation, or rejection of algorithmic starting points.
- Provide multiple ways of working with the same data for different needs across projects, or at different phases of the same project.
- Allow interactive (or near interactive) analysis with very large document collections (millions of documents), using only modest compute resources (such as an old laptop).
- Be accessible to a wide range of researchers.

# Background and Motivation

There are many existing approaches to text analytics


Relevant libraries:

- sklearn 
- gensim
- spacy
- quanteda
- bertopic?
- mallet
- (Should probably consider some corpus linguistic tools too)

problems:

- keep everything in memory
- simplistic models of what a document is (an array of features/Gensim/sklearn, a single text field + metadata/quanteda, text alone/spacy)
- ignore the presentation of documents necessary for the qualitative work of close reading
- make multimodal analysis (such as text and images) difficult
- require translation of documents to different software specific formats (search engines)
- require significant compute resources, or don't make effective use of modern multicore machines, or require finicky GPU hardware for performance


# Architecture



(Implementation note: this should describe the components and architecture, but only link to the detailed components for each.)

# Key Ideas and Components

## Assumptions

- A corpus contains a finite, enumerable set of documents that are uniquely identified by a key.
- 

Hyperreal makes no assumptions about what your document *is* - the interfaces only assume that it can be represented as a Python object in some way, and various transformations of that object are possible for display and analysis. This is intended so that documents can be left "where they are", and it should only be necessary to describe how to access and describe those documents.

## Interface



# Index


# FeatureClustering


# Next Phase High Level Architecture

*Note this is a work-in-progress document describing the **future** architecture. The software as implemented does not currently match this - as this design is implemented these notes will be migrated into the formal documentation.*

This should be read in conjunction with the [aims](wiki:project-aims) for guiding aims and principles. 

At a high level the most important components of Hyperreal are shown in the figure below. The `Corpus` and `Index` are the core components of Hyperreal, and are the fundamental units of using Hyperreal as a Python library. The Web server and Command Line Interface (CLI) provide additional interfaces but fundamentally are only calling the Python library: whatever you can do with one component you should be able to do with the others (possibly with a bit more work). 

<figure>

``` pikchr
Docs: [
  file  
  file at 0.1 se of previous fill white
  Top: file at 0.1 se of previous fill white
] 
text "Source docs" below at south of Docs

Python: [
  Corpus: box "Corpus"

  text "Document handling" below at south of Corpus

  Index: box "Index" at 1.5 e of Corpus.e
  arrow "indexed docs" "rendered docs" from Corpus.e to Index.w

  text "Inverted Index/Search Engine" below "Text Analytics" below at Index.s
] at 2.5 e of Docs.Top.e

PythonBox: box width Python.width height 1.2*Python.height at Python.c dashed

text "Python library" below at PythonBox.s

arrow "iterate" "retrieve" "transform" "render" from Docs.e to Python.Corpus.w

Server: box "Web Front End" at Python.Index.ne + (1.5,0.125) fit 
CLI: box "Command Line" "Interface" at Python.Index.se + (1.5,-0.125) fit

arrow from Python.Index.e to Server.w
arrow from Python.Index.e to CLI.w

```
<figcaption>The main components of the Hyperreal architecture.</figcaption>
</figure>

# Key Concepts and Components

Working from left to right in the diagram, the key concepts and components that make up Hyperreal are defined and explained in detail.

## The Corpus and the Document

A *corpus* object is used by Hyperreal to describe how to work with a specific collection of documents. A *document* is the fundamental unit of organisation for this software, but Hyperreal is designed to impose minimal constraints on what a document *is* and where it lives/how it is represented. The choice of what is a suitable document depends on the specific project and use case. For example, any of the following might be appropriate "documents" depending on the project: 

- a single sentence of text
- the text of an entire novel combined with metadata about the publication date and the author
- a social media post with text, an image, author profile information, and a set of hashtags
- the text of a speech in parliament with the date, speaker and political party of the speaker

You can think of the *corpus* object as describing a way to interact with a set of documents in a structured way: this means that your documents can stay where-ever they currently are, Hyperreal only needs a properly constructed *corpus* to understand how to access and transform them. Specifically the *corpus* object must describe a collection of documents by implementing an interface that enables:

1. Uniquely *identifying* all documents in a collection with a single key.
2. Incrementally *retrieving* documents matching specified keys (both in sequential batches and for random access).
3. *Transforming* documents into a specific form that can be used by the *Index*.
4. *Rendering* documents into specific formats, such as HTML for the [web interface](wiki:web-interface).
5. (Optionally) *Transforming and rendering* features extracted from documents, or rendering parts of whole documents.

Note that this set of requirements is particularly designed to enable very large collections of documents to be handled without needing everything to be loaded into memory at once: being able to retrieve only what's needed when it's needed means that you can do a lot more with a small machine like a laptop [^1].

[^1]: This design extends on some ideas [from Gensim's Corpora/Document API](https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html).

If you want to customise Hyperreal for your specific needs and documents, you will need to [implement a custom *corpus*](wiki:custom-corpus-howto), possibly starting with one of the [builtin corpora](wiki:supported-corpora) as a reference. Note also that this means that the *corpus* is the main point of customisation relating to documents and how they are displayed, including in through the Web Frontend - there shouldn't be much need for customisation of other parts of Hyperreal.

## The Index

The *index* is the main API for making Hyperreal do anything interesting: it is the fundamental object you will interact with for most functionality, including the advanced text analytics functionality. This is also the main object that the Web Frontend and command line interfaces use for working with Hyperreal.

The core functionality that the *index* provides is to create an [inverted index](https://en.wikipedia.org/wiki/Inverted_index) to support Boolean full text search on the documents described by the *corpus*. This inverted index and associated search functionality is then used by the index

Need a note about the bitmap of doc_ids as the core representation and unit of computation - ie concrete implementation details of the inverted index (and why the need for the fast bitmap operations was why this was an intentional design choice/reason not to use another off the shelf search engine).

1. Index documents as described by the *corpus*.
2. Retrieve sets of documents matching a specific feature
3. Enable composition of queries from different features.

### Indexed document format

As mentioned earlier, Hyperreal tries not to impose requirements on what a document is, but we do impose requirements on what the indexed form of a document is.

An example of the form of a document (represented by a Python tuple representing some text and a DateTime), and one indexed form of that document is shown below.

```json
doc = (1, 'The cat sat on the mat.', datetime(2024, 1, 1, 12, 30))

indexed_doc = (
    1, 
    {
        'text': ['the', 'cat', 'sat', 'on', 'the', 'mat'],
        'date': datetime(2024, 1, 1)
    }
)

```

The inverted index converts the above document (implicitly) into a mapping of features (field, value) pairs, to the document id.

```
{
    ('text', 'the') : {1},
    ('text', 'cat') : {1},
    ...
    ('date', '2024-01-01T00:00:00+00:00') : {1},

}
```


The form and type signature of an Indexable form of a document are:

1. The indexable is a mapping from string fields to values.
2. Field values can either be:
   - A single value
   - A list of values, indicating that the field has multiple _ordered_ values. This is the appropriate choice for fields with positional/ordering based information such as tokenised text.
   - A set of values (representing that a field has multiple values for a document, but no particular ordering.)
3. Values must be hashable.
4. Values must be represented as a single value in an SQLite database.

At index time the multiplicity of each field will be recorded, and this will be used to enable particular features.

A combination of a field and a discrete value is a *feature* (for now - this might change or be extended in the future to accomodate some additional needs).

### Customising Feature Representation

The indexed data format is stored as is in the inverted index. Without any customisation any values are retrieved and displayed using whatever there stored representation is - so text will be round-tripped as text, datetime objects will be implicitly converted to strings for storage and so on.

To customise this behaviour the corpus object can be extended to describe field specific rendering in different contexts. This is done by ... - for example to change how dates are rendered the ... is added to the corpus???




