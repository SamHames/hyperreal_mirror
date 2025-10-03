"""
HTML generating functions for the web viewer.

This is primarily about the broader structure of generating complete HTML pages - what
is passed in to this is primarily self-rendering objects from other parts of Hyperreal.

"""

import math
from urllib.parse import quote

from tinyhtml import frag, h, html, raw


def render_feature_stats_table(
    feature_stats,
    caption=None,
    select_form_id=None,
):

    if caption is not None:
        caption_elem = h("caption")(caption)

    headers = [
        "Field",
        "Value",
        "Docs",
    ]

    if select_form_id:
        header_fields.append("Select")

    header_fields = [h("th", scope="col")(header_text) for header_text in headers]

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

        if "feature_url" in stats:
            html_value = h("a", href=stats["feature_url"])(html_value)

        cells.append(h("th", scope="row", klass="feature-value")(html_value))

        cells.append(h("td")(stats.get("doc_count")))

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


def render_features_dl(feature_stats, select_form_id=None):

    last_field = None

    elements = []

    for i, (feature, stats) in enumerate(feature_stats.items()):

        field = feature[0]

        if field != last_field:
            elements.append(h("dt", klass="field")(field))
            last_field = field

        html_values = feature[1:]

        if len(html_values) == 2:
            html_value = (html_values[0], "-", html_values[1])
        else:
            html_value = html_values[0]

        display_value = html_value
        if "feature_url" in stats:
            display_value = h("a", href=stats["feature_url"])(display_value)

        selector = ""
        if select_form_id and stats.get("select_form_value"):
            selector = h(
                "input",
                type="checkbox",
                name="f",
                form=select_form_id,
                value=stats["select_form_value"],
            )

        elements.append(h("dd", klass="feature-value cluster")(display_value, selector))

    return h("dl", klass="feature-list cluster")(elements)


def render_feature_clustering(
    feature_clustering,
    cluster_stats,
    select_form_id=None,
):

    clusters = []

    for cluster_id, stats in cluster_stats.items():

        items = []

        title = h("h2", id=f"cluster-{cluster_id}")(
            h("a", href=stats["feature_url"])(f"Cluster {cluster_id}")
        )

        selector = ""
        if select_form_id is not None:
            selector = h(
                "input",
                type="checkbox",
                name="c",
                form=select_form_id,
                value=cluster_id,
            )

        items.append(
            h("div", klass="feature-value cluster")(
                title,
                selector,
            )
        )

        features = feature_clustering[cluster_id]

        items.append(render_features_dl(features, select_form_id=select_form_id))

        footer = None
        if stats.get("expand_url", False):
            remaining_count = stats["matching_feature_count"] - len(features)
            footer = h("div", klass="expand-url")(
                h(
                    "a",
                    href=stats["expand_url"],
                )("+ ", remaining_count, " more")
            )

        items.append(footer)

        clusters.append(h("li")(h("div", klass="stack feature-cluster")(items)))

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


def cluster_navigation(browse_url, cluster_ids, selected=None):
    """Generate a selector to jump to any cluster."""

    return h("form", method="get", action=browse_url)(
        h("select", name="c")(
            [h("option", selected=cluster_id == selected, value=cluster_id)(cluster_id)]
            for cluster_id in cluster_ids
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

    column_width_style = None
    if body_columns and len(body_columns) == 1:
        column_width_style = f"--column-width: 100%"

    if sub_nav_links:
        body_header = [sub_nav, body_header]

    columns = [h("div", klass="column stack")(column) for column in body_columns]

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
                h("div", klass="columns", style=column_width_style)(columns),
            ),
        ),
    )


def column_content_with_header(title, content, content_klass="scrollable"):
    return (h("h1")(title), h("div", klass=content_klass)(content))


def render_field_table(index_summary):

    header = ["Field"] + list(next(iter(index_summary.values())).keys())

    return h("table")(
        h("caption")("Overview of the indexed fields for this collection."),
        h("thead")(h("tr")([h("th", scope="col")(col_name) for col_name in header])),
        h("tbody")(
            [
                h("tr")(
                    h("th", scope="row")(key),
                    *[h("td")(item) for item in stats.values()],
                )
                for key, stats in index_summary.items()
            ]
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
    --s-4: calc(var(--s-3) / var(--ratio));
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
    overflow: hidden;
    padding: 0 var(--s-1);
    --space: var(--s0);
    max-width: var(--column-width);
    scrollbar-color: black white;
    flex: 1;
}

.column > * {
    padding: 0 var(--s-2);
}

.column h1 {
    box-shadow: var(--s-3) var(--s-3) var(--s-3) 0 var(--border-color);
    background-color: oklch(90% 0 0);
}

.scrollable {
    overflow-y: scroll;
    flex: 1;
    padding: var(--s-3);
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

.flex-no-grow {
    flex: 0 1 auto;
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
    grid-template-columns: auto auto auto;
    padding: var(--s-1);
}

.feature-table :is(tbody, thead) {
    border-bottom: var(--thin) solid black;
    display: grid;
    grid-template-columns: subgrid;
    grid-column: span 3;
    gap: 0;
}

.feature-table tr {
    display: grid;
    grid-template-columns: subgrid;
    grid-column: span 3;
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

.feature-list {
    flex: 1;
    gap: var(--s-1);
}

.feature-list a {
    text-decoration: none;   
}

.feature-value {
    gap: var(--s-3);
}

.feature-field dt::after {
    content: ':';
}

.feature-clustering > li {
    padding-left: var(--s-3);
}

.feature-cluster {
    --space: var(--s-2);
}

.expand-url {
    text-align: right;
}

.feature-value:has(input:checked){
    background-color: yellow;
}

/*************/
"""
