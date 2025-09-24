"""
HTML generating functions for the web viewer.

This is primarily about the broader structure of generating complete HTML pages - what
is passed in to this is primarily self-rendering objects from other parts of Hyperreal.

"""

import math
from urllib.parse import quote

from tinyhtml import frag, h, html, raw


def heatmap_legend(label, start, stop, steps):
    """
    Make a legend for the heatmap scales.

    """

    width = stop - start
    bin_width = width / steps

    bins = []

    for i in range(steps + 1):
        step = start + bin_width * i
        bins.append(h("td", klass="heatmap", style=f"--sim: {step:.3f}")(f"{step:.1f}"))

    return h("table")(
        h("caption")("Legend for similarity score heatmap."),
        h("tr", klass="cluster legend")(h("th", scope="col")(label), *bins),
    )


def heatmap_cells(stats):
    """
    Render the heatmap cell entries as td.

    """
    cells = []

    ##### Hits/query similarity cells
    hits = stats.get("hits", None)
    similarity = stats.get("jaccard_similarity", None)

    if hits is not None:
        rounded_sim = f"{similarity:.3f}"
        style = f"--sim: {rounded_sim}"

        if hit_count_url := stats.get("hit_count_url", None):
            hits = h("a", href=hit_count_url)(hits)

        cells.append(h("td", klass="heatmap", style=style)(hits))
        cells.append(h("td", klass="invisible")(rounded_sim))
    else:
        cells.append(h("td")(""))
        cells.append(h("td", klass="invisible")(""))

    ##### Docs/relative frequency cells
    docs = stats["doc_count"]
    rel_docs = stats["relative_doc_count"]

    if doc_count_url := stats.get("doc_count_url", None):
        docs = h("a", href=doc_count_url)(docs)

    rounded_rel_docs = f"{rel_docs:.3f}"
    style = f"--sim: {rounded_rel_docs}"

    cells.append(h("td", klass="heatmap", style=style)(docs))
    cells.append(h("td", klass="invisible")(rounded_rel_docs))

    return cells


def render_feature_stats_table(
    feature_stats,
    caption=None,
    select_form_id=None,
):

    if caption is not None:
        caption_elem = h("caption")(caption)

    header_fields = [
        h("th", scope="col")(header_text)
        for header_text in (
            "Field",
            "Value",
            "Hits",
            "Query Similarity",
            "Docs",
            "Relative Docs",
            "Select" if select_form_id else "",
        )
    ]

    header_fields[3] = h("th", scope="col", klass="invisible")("Query Similarity")
    header_fields[5] = h("th", scope="col", klass="invisible")("Relative Docs")

    header = h("thead")(h("tr")(header_fields))
    body_rows = []

    for feature, stats in feature_stats.items():

        cells = []

        ##### Field and value cells

        field = feature[0]

        cells.append(h("th", scope="row", klass="feature-field")(field))

        html_values = feature[1:]

        if len(html_values) == 2:
            html_value = (html_values[0], "-", html_values[1])
        else:
            html_value = html_values[0]

        cells.append(h("th", scope="row", klass="feature-value")(html_value))

        cells.extend(heatmap_cells(stats))

        selector = ""
        if select_form_id and stats.get("select_form_value"):
            selector = h(
                "input",
                type="checkbox",
                name="f",
                form=select_form_id,
                value=stats["select_form_value"],
            )

        cells.append(h("td")(selector))

        body_rows.append(h("tr")(cells))

    return h("table", klass="feature-table")(caption, header, h("tbody")(body_rows))


def render_feature_clustering(
    feature_clustering,
    cluster_stats,
    select_form_id=None,
):

    clusters = []

    for cluster_id, stats in cluster_stats.items():

        header_fields = [
            h("th", scope="col")(header_text)
            for header_text in (
                "Cluster",
                "Matched Features",
                "Total Features",
                "Hits",
                "Query Similarity",
                "Docs",
                "Relative Docs",
                "Select" if select_form_id else "",
            )
        ]

        header_fields[4] = h("th", scope="col", klass="invisible")("Query Similarity")
        header_fields[6] = h("th", scope="col", klass="invisible")("Relative Docs")

        cells = []

        cells.append(h("th", scope="row")(cluster_id))

        features = feature_clustering[cluster_id]

        display_feature_count = len(features)
        matching_feature_count = stats["matching_feature_count"]

        cells.append(h("td")(matching_feature_count))
        cells.append(h("td")(stats["feature_count"]))

        cells.extend(heatmap_cells(stats))

        selector = ""
        if select_form_id is not None:
            selector = h(
                "input",
                type="checkbox",
                name="c",
                form=select_form_id,
                value=cluster_id,
            )

        cells.append(h("td")(selector))

        header = h("table", klass="cluster-stats-table", id=f"cluster-{cluster_id}")(
            h("thead")(h("tr")(header_fields), h("tbody")(cells))
        )

        footer = None
        if stats.get("expand_url", False):
            footer = h("div")(
                h(
                    "a",
                    href=stats["expand_url"],
                )("Show all ", matching_feature_count, " matching features")
            )

        clusters.append(
            h("li", klass="cluster-features")(
                header,
                render_feature_stats_table(features, select_form_id=select_form_id),
                footer,
            )
        )

    return h("ol", klass="stack feature-clustering")(clusters)


def render_feature_edit_forms(
    add_to_query, create_action, merge_action, new_query, current_query
):

    query_input = None
    if current_query is not None:
        query_input = h("input", type="hidden", name="query", value=current_query)

    return h(
        "form", klass="stack", id="edit-model-form", method="post", action=create_action
    )(
        query_input,
        h("button", type="submit")("Create cluster from selected features"),
        h("button", type="submit", formaction=merge_action)("Merge selected clusters"),
        h("button", type="submit", formmethod="get", formaction=add_to_query)(
            "Create query clause from selected"
        ),
        h("button", type="submit", formmethod="get", formaction=new_query)(
            "Start new query with selected"
        ),
    )


def cluster_navigation(browse_url, cluster_ids):
    """Generate a selector to jump to any cluster."""

    return h("form", method="get", action=browse_url)(
        h("select", name="c")(
            [h("option", value=cluster_id)(cluster_id)] for cluster_id in cluster_ids
        ),
        h("button", type="submit")("Go to cluster"),
    )


def generate_search(search_url, search_fields):
    return h("form", method="get", action=search_url)(
        h("select", name="search-field")(
            h("option", value=field)(field) for field in search_fields
        ),
        h("input", type="text", name="search-value")(),
        h("button", type="submit")("Search"),
    )


def generate_nav(label, links, klass=None):
    """Generate a navigation element."""

    # Make sure this is a valid HTML id
    nav_label = "-".join(label.split())

    nav_id = f"nav-{nav_label}"
    label = h("span", klass="nav-label", id=nav_id)(label)
    return h("nav", aria_labelled_by=nav_id)(
        label,
        h("div", klass="inlined")(
            h("ul", klass="cluster")(
                h("li")(h("a", href=href)(link_text) if href else link_text)
                for link_text, href in links
            ),
        ),
    )


def search_results_header(sample_doc_count, matching_doc_count):
    return f"{sample_doc_count} of {matching_doc_count} matching documents"


def full_page(
    page_title,
    body_columns,
    body_header=None,
    column_flex=None,
    sub_nav_links=None,
    sub_nav_label=None,
    extra_css=None,
):
    """Render a complete page with navigation and a page title."""

    nav_links = [
        ("Index Overview", "/"),
        ("Browse", "/browse/"),
    ]
    main_nav = generate_nav("Main", nav_links)

    sub_nav_links = sub_nav_links or {}
    sub_nav = [
        generate_nav(sub_nav_label, sub_nav_links)
        for sub_nav_label, sub_nav_links in sub_nav_links.items()
    ]

    extra_css = extra_css or ""

    column_flex = column_flex or {}

    column_width_style = None
    if body_columns and len(body_columns) == 1:
        column_width_style = f"--column-width: 100%"

    if sub_nav_links:
        body_header = [sub_nav, body_header]

    return html(lang="en")(
        h("head")(
            h("meta", name="viewport", content="width=device-width, initial-scale=1"),
            h("title")(page_title),
            h("style")(raw(default_css), raw(extra_css)),
        ),
        h("body")(
            h("header", klass="bordered")(main_nav),
            h("main")(
                (
                    h("div", klass="main-header cluster bordered")(body_header)
                    if body_header
                    else None
                ),
                h("div", klass="columns", style=column_width_style)(
                    (
                        h(
                            "div",
                            klass="column bordered",
                        )(h("h1")(col_title), col)
                        for col_title, col in body_columns
                    ),
                ),
            ),
        ),
    )


def render_field_table(index_summary):

    header = ["Field"] + list(next(iter(index_summary.values())).keys())

    return h("table")(
        h("caption")("Overview of the indexed fields for this collection."),
        h("thead")(h("tr")(h("th", scope="col")(col_name) for col_name in header)),
        h("tbody")(
            (
                h("tr")(
                    h("th", scope="row")(key),
                    (h("td")(item) for item in stats.values()),
                )
                for key, stats in index_summary.items()
            )
        ),
    )


# This is the default CSS.
# A corpus can have additional CSS, it will be appended after this to allow it to
# override/customise any default rules specified here.
default_css = """
/* Box sizing everywhere */
*,
*::before,
*::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

:root {
    --ratio: 1.618;
    --s0: 1rem;
    --s1: calc(var(--s0) * var(--ratio));
    --s2: calc(var(--s1) * var(--ratio));
    --s3: calc(var(--s2) * var(--ratio));
    --s-1: calc(var(--s0) / var(--ratio));
    --s-2: calc(var(--s-1) / var(--ratio));
    --s-3: calc(var(--s-2) / var(--ratio));
    --space: var(--s-1);
    --column-width: 72ch;
    --border-color: oklch(50% 0 0);
    --thin: 0.1rem;
}

.cluster {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space);
}

header {
    margin: var(--s0);
    height: fit-content;
    padding: var(--s-1);
}

.bordered {
    border: var(--thin) solid var(--border-color);
}

nav ul {
    list-style: none;
    margin-bottom: var(--s-1);
}

.nav-label {
    margin-inline-end: var(--s-1);
}

.inlined {
    display: inline-block;
}

ul > li {
    list-style: none;
}

body {
    display: flex;
    flex-direction: column;
    height: 100vh;
}

main {
    display: flex;
    flex-direction: column;
    min-height: 0;
}

main > * {
    margin: var(--s0);
}

.main-header {
    margin: var(--s0);
    height: fit-content;
    padding: var(--s-1);
}

.columns {
    overflow: hidden;
    gap: var(--s1);
    display: flex;
    flex-direction: row;
    justify-content: center;
}

.column {
    overflow-y: scroll;
    padding: 0 var(--s-1);
    max-width: var(--column-width);
    scrollbar-color: black white;
    flex: 1;
}


h1 {
    font-size: 120%;
}

h2, h3 {
    font-size: 100%;
}

.stack {
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
}

.stack > * {
  margin-block: 0;
}

.stack > * + * {
  margin-block-start: var(--space, 1rem);
}

.concordance-container {
    width: 100%;
    display: inline-block;
}

/* Concordance handling CSS */
.concordance {
    width: 100%;
    white-space: nowrap;
    display: grid;
    grid-template-columns: auto auto auto;
    list-style: none;
}

.concordance-line {
    display: grid;
    grid-template-columns: subgrid;
    grid-column: span 3;
    width: 100%;
}

.concordance-line > * {
    text-overflow: ellipsis;
    overflow: clip;
    min-width: 0;
}

.concordance-match {
    width: min-content;
    padding: 0 var(--s-3);
    text-align: center;
}

.concordance-pre {
    text-align: right;
    direction: rtl;
    width: auto;
}

.concordance-post {
    text-align: left;
    width: auto;
}

.matches summary {
    cursor: pointer;
}

.matches details summary > * {
    display: inline;
}

.matches dt::after {
    content: ":";
}

@view-transition {
  navigation: replace;
}

.search-hit > * {
    --space: var(--s-1);
}

.search-hit {
    --space: var(--s1);
}

.legend {
    gap: 0
}

.legend :is(td, th) {
    padding: var(--s-2);
    margin: 0;
}

.field::after {
    content: ": ";
}

.query {
    --space: var(--s-3);
}

.query-operator {
    text-align: center;
    font-style: italic;
}

/****** Layout for feature tables *******/

.feature-table {
    width: 100%;
    display: grid;
    grid-template-columns: auto auto auto 0 auto 0 auto;
    padding: var(--s-1);
}

.feature-table :is(tbody, thead) {
    border-bottom: var(--thin) solid black;
    display: grid;
    grid-template-columns: subgrid;
    grid-column: span 7;
    gap: 0;
}

.feature-table tr {
    display: grid;
    grid-template-columns: subgrid;
    grid-column: span 7;
}

.feature-table :is(td, th) {
    padding: var(--s-3);
    overflow-x: clip;
    text-overflow: ellipsis;
    font-family: monospace, monospace;
    white-space: nowrap;
    min-width: 0;
}


.feature-table th {
    text-align: left;
}

.feature-table td {
    text-align: right;
}

.cluster-stats-table {
    display: grid;
    grid-template-columns: auto auto auto auto 0 auto 0 auto;
}

.cluster-stats-table :is(thead, tbody) {
    border-bottom: var(--thin) solid black;
    display: grid;
    grid-template-columns: subgrid;
    grid-column: span 8;
    gap: 0;
}

.cluster-stats-table tr {
    display: grid;
    grid-template-columns: subgrid;
    grid-column: span 8;
}

.cluster-stats-table :is(td, th) {
    padding: var(--s-3);
    overflow-x: clip;
    text-overflow: ellipsis;
    font-family: monospace, monospace;
    min-width: 0;
}

.cluster-stats-table td {
    text-align: right;
}

.cluster-stats-table td:last-child {
    text-align: center;
}

.heatmap, .heatmap a, .heatmap a:visited  {
    background: oklch(100% calc(sqrt(var(--sim, 0))*80%) 20);
    color: black;
}

.invisible {
    font-size: 0;
    width: 0;
    max-width: 0;
    padding: 0 !important; 
}

.feature-table td:last-child {
    text-align: center;
}

.feature-clustering {
    list-style: none;
}

:is(tr, .cluster-stats-table):has(input:checked){
    border: var(--s-3) solid yellow;
}

/*************/
"""
