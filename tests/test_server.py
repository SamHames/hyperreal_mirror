"""
Tests for the server module.

"""

import concurrent.futures as cf
import multiprocessing as mp
import pathlib
import shutil
import uuid

import cherrypy
import pytest
import requests
from lxml import html

import hyperreal

servers = [
    # Index server with no associated corpus.
    (
        None,
        hyperreal.corpus.EmptyCorpus,
        pathlib.Path("tests", "index", "alice_index.db"),
    ),
    # PlainTextCorpus
    (
        pathlib.Path("tests", "corpora", "alice.db"),
        hyperreal.corpus.PlainTextSqliteCorpus,
        pathlib.Path("tests", "index", "alice_index.db"),
    ),
]


@pytest.fixture(params=servers, name="server_url")
def fixture_server(tmp_path, request):
    """Server fixture that handles single file corpora."""
    source_corpus_path, source_corpus_class, source_index = request.param

    if source_corpus_path is not None:
        corpus_path = tmp_path / str(uuid.uuid4())
        shutil.copy(source_corpus_path, corpus_path)
    else:
        corpus_path = None

    index_path = tmp_path / str(uuid.uuid4())
    shutil.copy(source_index, index_path)

    context = mp.get_context("spawn")

    with cf.ProcessPoolExecutor(4, mp_context=context) as pool:
        if corpus_path is not None:
            corp = source_corpus_class(corpus_path)
        else:
            corp = hyperreal.corpus.EmptyCorpus()

        idx = hyperreal.index.Index(index_path, corpus=corp, pool=pool)

        features = [row[0] for row in idx.field_features("text", min_docs_count=1)]
        clustering = hyperreal.cluster_features(
            idx,
            features,
            16,
            iterations=10,
        )

        for cluster in clustering.values():
            idx.create_cluster_from_features(cluster)

        index_server = hyperreal.server.SingleIndexServer(index_path, pool=pool)
        engine = hyperreal.server.launch_web_server(index_server, port=0)
        host, port = cherrypy.server.bound_addr
        yield f"http://{host}:{port}"
        engine.exit()


def test_server(server_url):
    """
    Start an index only server in the background using the CLI.

    Returns the local host and port of the test server.

    """

    r = requests.get(server_url, timeout=1)
    r.raise_for_status()

    # There should be an index listing in there somewhere
    doc = html.document_fromstring(r.content)
    links = [l.attrib["href"] for l in doc.findall(".//a")]

    assert "/index/0" in links

    r = requests.get(server_url + "/index/0", timeout=1)
    r.raise_for_status()

    doc = html.document_fromstring(r.content)
    links = {l.attrib["href"] for l in doc.findall(".//a")}
    must_be_present = {
        "/index/0/cluster/1",
        "/index/0/?cluster_id=1",
        "/index/0/details",
    }

    assert len(links & must_be_present) == len(must_be_present)

    for l in links:
        if l.startswith("/index/0/?f="):
            must_be_present.add(l)
            break
    else:
        raise ValueError("Missing a feature link")

    for test_link in must_be_present:
        r = requests.get(server_url + test_link, timeout=1)
        r.raise_for_status()


# Test details

# Test cluster creation page opens

# Test cluster creation works for one or more fields, and that they render
# correctly afterwards.

# Test a sample of links work - random walk through the whole page.

# Test deleting all wipes everything.
