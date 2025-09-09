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
        bins.append(h("td", klass="heatmap", style=f"--w: {step:.3f}")(f"{step:.1f}"))

    return h("table")(
        h("caption")("Legend for similarity score heatmap."),
        h("tr", klass="cluster legend")(h("th", scope="col")(label), *bins),
    )


def render_feature_stats_table(
    feature_stats,
    caption=None,
    feature_url_key=None,
    count_stat=None,
    heatmap_stat=None,
    feature_form_details=(None, None),
):

    if caption is not None:
        caption_elem = h("caption")(caption)

    header_fields = [
        h("th", scope="col", klass="feature-field")("Field"),
        h("th", scope="col", klass="feature-value")("Value"),
    ]
    if count_stat:
        header_fields.append(h("th", scope="col", klass="display-count")(count_stat))
    if heatmap_stat:
        header_fields.append(
            # style might be set from the cluster level similarity, so need to reset
            h("th", scope="col", klass="heatmap-value", style="--w: 0;")(heatmap_stat)
        )
    if feature_form_details[0] is not None:
        header_fields.append(h("th", scope="col")("Select"))

    header = h("thead")(h("tr")(header_fields))
    body_rows = []
    last_field = None

    for feature, stats in feature_stats.items():

        cells = []

        field = feature[0]

        klass = "feature-field"
        if field == last_field:
            klass = "feature-field repeat-in-run"

        cells.append(h("th", scope="row", klass=klass)(field))

        html_values = feature[1:]

        if len(html_values) == 2:
            html_value = (html_values[0], "-", html_values[1])
        else:
            html_value = html_values[0]

        if feature_url_key is not None:
            href = stats[feature_url_key]
            feature_value = h("a", href=href)(html_value)
        else:
            feature_value = html_value

        cells.append(h("th", scope="row", klass="feature-value")(feature_value))

        # Show a count if present
        if count_stat is not None:
            cells.append(h("td", klass="display-count")(stats[count_stat]))

        # Show a cell that will by default only show the heatmap colour, not the
        # actual count.
        style = None
        if heatmap_stat is not None:
            heatmap_value = stats[heatmap_stat]
            style = f"--w: {heatmap_value:.3f};"
            cells.append(h("td", klass="heatmap-value")(heatmap_value))

        if feature_form_details[0] is not None and feature_form_details[1] is not None:
            selector = h(
                "input",
                type="checkbox",
                name="feature",
                form=feature_form_details[0],
                value=stats[feature_form_details[1]],
            )
            cells.append(h("td")(selector))

        body_rows.append(h("tr", klass="heatmap", style=style)(cells))

        last_field = field

    return h("table", klass="feature-table")(caption, header, h("tbody")(body_rows))


def render_feature_clustering(
    feature_clustering,
    cluster_stats,
    count_stat=None,
    heatmap_stat=None,
    feature_url_key=None,
    header_url_key=None,
    seemore_url_key=None,
    feature_form_details=(None, None),
):

    clusters = []

    for cluster_id, stats in cluster_stats.items():

        features = feature_clustering[cluster_id]

        display_feature_count = len(features)
        matching_feature_count = stats["matching_feature_count"]

        footer = None
        if seemore_url_key is not None and stats.get(seemore_url_key, False):
            footer = h("div")(
                h(
                    "a",
                    href=stats[seemore_url_key],
                )("Show all ", matching_feature_count, " matching features")
            )

        selector = None
        if feature_form_details[0] is not None:
            selector = h(
                "input",
                type="checkbox",
                name="c",
                form=feature_form_details[0],
                value=cluster_id,
            )

        style = None
        if heatmap_stat is not None:
            style = f"--w: {stats[heatmap_stat]:.3f}"

        cluster_title = h("h2", id=f"cluster-{cluster_id}", klass="cluster spread")(
            "Cluster: ", cluster_id, selector
        )

        if header_url_key is not None:
            cluster_title = h("a", href=stats[header_url_key])(cluster_title)

        header = h("div", klass="feature-table-header")(
            cluster_title,
        )

        clusters.append(
            h("li", klass="heatmap-left cluster-features", style=style)(
                header,
                render_feature_stats_table(
                    features,
                    feature_url_key=feature_url_key,
                    count_stat=count_stat,
                    heatmap_stat=heatmap_stat,
                    feature_form_details=feature_form_details,
                ),
                footer,
            )
        )

    return h("ol", klass="stack feature-clustering")(clusters)


def render_feature_edit_forms(create_action, merge_action):

    return h("form", id="edit-model-form", method="post", action=create_action)(
        h("button", type="submit")("Create cluster from selected features"),
        h("button", type="submit", formaction=merge_action)("Merge selected clusters"),
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
        ("Browse", "/browse"),
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
            h("header")(main_nav),
            h("main")(
                (
                    h("div", klass="main-header cluster")(body_header)
                    if body_header
                    else None
                ),
                h("div", klass="columns", style=column_width_style)(
                    (
                        h(
                            "div",
                            klass="column",
                            style=f"--column-flex: {column_flex.get(col_index, 1)}",
                        )(col)
                        for col_index, col in enumerate(body_columns)
                    ),
                ),
            ),
        ),
    )


def list_search_results(search_results, sample_doc_count=None, matching_doc_count=None):

    header_label = "Sample of matching docs"
    if sample_doc_count is not None and matching_doc_count is not None:
        header_label = f"{sample_doc_count} of {matching_doc_count} matching documents"
    elif sample_doc_count is not None:
        header_label = f"{sample_doc_count} sample documents"
    elif matching_doc_count is not None:
        header_label = f"Sample of {matching_doc_count} matching_documents"

    return (
        h("div", klass="header")(
            h("h2")(header_label),
        ),
        search_results,
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
    --column-width: 72ch;
}

.cluster {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-1);
}

header {
    margin: var(--s0);
    height: fit-content;
    border: solid;
    padding: var(--s-1);
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
    border: solid black;
    margin: var(--s0);
    height: fit-content;
    border: solid;
    padding: var(--s-1);
}

.columns {
    overflow: hidden;
    gap: var(--s1);
    display: flex;
    flex-direction: row;
    flex: 1;
    justify-content: center;
}

.column {
    overflow-y: scroll;
    flex: var(--column-flex, 1);
    border: solid;
    padding: var(--s0);
    max-width: var(--column-width);
}

pre {
    white-space: pre-wrap;
}

h1, h2, h3 {
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

/* Concordance handling CSS */
.concordance {
    --space: var(--s-2);
    margin: var(--s-2);
    width: 100%;
    table-layout: fixed;
}

.concordance td {
    text-overflow: ellipsis;
    overflow: clip;
    white-space: nowrap;
}

.concordance-match {
    width: min-content;
    padding: 0 var(--s-3);
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

.matches summary,details {
    display: inline;
    cursor: pointer;
}

.matches > * {
    margin-inline-end: var(--s-1);
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
    padding: var(--s-1);
    margin: 0;
}

/****** Layout for feature tables *******/

.cluster-features > * {
    padding: var(--s-3);
}

.feature-table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
}

.feature-table thead {
    whitespace: nowrap;
    border-bottom: var(--s-3) double black;
}

.feature-table thead th {
    overflow-x: clip;
    text-overflow: ellipsis;
}

.feature-table :is(td, th) {
    padding: var(--s-3);
    overflow-x: clip;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-family: monospace, monospace;
}

.feature-value {
    text-align: end;
    width: 50%;
}

.feature-field {
    width: auto;
}

tbody :is(.feature-field, .feature-value) {
    font-weight: normal;
}

.repeat-in-run {
    font-size: 60%;
    font-weight: lighter;
}

.heatmap {
    background: oklch(calc(1 - var(--w, 0)) 0 0);
    color: oklch(
        calc(round(var(--w, 1) + 0.15))
        0 0
    );
}

.heatmap-value {
    font-size: 0;
    width: var(--s1);
}

.heatmap a, .heatmap a:visited {
    color: oklch(
        calc(round(var(--w, 1) + 0.15))
        0
        0
    );
}

.heatmap-left {
    border-left: var(--s1) solid oklch(calc(1 - var(--w, 0)) 0 0);
}

.display-count {
    text-align: end;
    font-family: monospace, monospace;
    width: auto;
}

.feature-clustering {
    list-style: none;
}

:is(tr, .feature-table-header):has(input:checked){
    outline: solid yellow;
}



/*************/
"""
