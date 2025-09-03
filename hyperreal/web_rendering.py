"""
HTML generating functions for the web viewer.

This is primarily about the broader structure of generating complete HTML pages - what
is passed in to this is primarily self-rendering objects from other parts of Hyperreal.

"""

import math
from urllib.parse import quote

from tinyhtml import frag, h, html, raw

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
    --header: #efefef;
}

.cluster {
    display: flex;
    flex-wrap: wrap;
}

header {
    padding: var(--s0);
    background-color: var(--header);
    height: fit-content;
    border-bottom: solid;
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
    margin-right: var(--s-1);
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

.columns {
    overflow: hidden;
    gap: var(--s3);
    display: flex;
    flex-direction: row;
    flex: 1;
}

.column {
    overflow: scroll;
    flex: var(--column-flex, 1);
}

pre {
    white-space: pre-wrap;
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

.stat-row {
    display: flex;
    justify-content: end;
    gap: var(--s-1);
    flex-wrap: nowrap;
    align-items: center;
    padding: 0 var(--s-3);
}

.stat-row :not(:first-child) {
    flex-shrink: 0;
}

.stat-row :first-child {
    text-align: left;
    text-overflow: ellipsis;
    overflow: hidden;
}

:is(.header, .stat-row):has(>input:checked){
    border: solid yellow;
}

.feature-list {
    margin-block-end: var(--s0);
}

.feature-list a {
    text-decoration: none;
    display: inline-block;
}

.feature-list dt::after {
    content: ":";
}

.area-mark {
    height: 1rem;
    width: 1rem;
    text-align: center;
    display:flex;
    justify-content:center;
    align-items:center;
}

.area-mark::before {
    content: "";
    height: var(--w);
    width: var(--w);
    margin: 0 auto;
    background: black;
    display: inline-block;   
    vertical-align: middle;
}

.feature-clustering {
    line-height: 130%;
    gap: var(--s0);
    align-items: stretch;
    justify-content: space-between;
}

.feature-clustering > * {
    --space: var(--s-3);
    overflow-y: clip;
    overflow-x: clip;
    white-space: nowrap;
    flex: 1 auto;
    max-width: 100%;
}

.header {
    background: var(--header);
    text-align: right;
    border-bottom: solid;
    margin-block-end: var(--s-2);
    position: sticky;
    top: 0;
    padding: var(--s-3);
}

.header h2 {
    display: inline-block;
    font-size: 100%;
}

.display-number {
    font-family: monospace;
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

"""


def calculate_area_mark(normalised_area, total_doc_count, min_mapping_size=0.05):
    """
    Transform a normalised value in range [0, 1] to a renderable unit in HTML.

    This transform maps 0, 1 -> 0, 1, and maps 1/total_doc_count -> 0.1. This is
    intended to enable rendering a wide range of values as areas.

    """
    power = math.log10(min_mapping_size) / (-math.log10(total_doc_count))
    return normalised_area**power


def render_feature_stats_as_dl(
    feature_stats,
    feature_order=None,
    feature_url_key=None,
    display_stat=None,
    area_stat=None,
    total_doc_count=None,
    klass="stack feature-list",
    feature_form_id=None,
    feature_form_key=None,
):

    feature_order = feature_order or feature_stats.keys()

    items = []
    last_field = None

    for feature in feature_order:
        field = feature[0]
        html_value = feature[1:]

        if len(html_value) == 2:
            html_value = (html_value[0], "/", html_value[1])

        details = feature_stats[feature]

        if field != last_field:
            items.append(h("dt")(h("em")(field)))

        selector = None
        if feature_form_id is not None and feature_form_key is not None:
            selector = h(
                "input",
                type="checkbox",
                name="feature",
                value=details[feature_form_key],
                form=feature_form_id,
            )

        area_mark = None
        if area_stat is not None:
            area_side = calculate_area_mark(details[area_stat], total_doc_count)
            style = f"--w: {area_side:.3f}rem;"
            area_mark = h("div", style=style, klass="area-mark")()

        display = None
        if display_stat is not None:
            display = h("span", klass="display-number")(details[display_stat])

        if feature_url_key is not None:
            href = details[feature_url_key]
            value = h("a", klass="display-number", href=href)(html_value)
        else:
            value = h("span", klass="display-number")(html_value)

        items.append(h("dd", klass="stat-row")(value, display, area_mark, selector))

        last_field = field

    return h("dl", klass=klass)(items)


def render_feature_clustering(
    clustering,
    cluster_stats,
    total_doc_count,
    cluster_order=None,
    area_stat="relative_doc_count",
    display_stat=None,
    feature_url_key=None,
    header_url_key=None,
    seemore_url_key=None,
    feature_form_id=None,
    feature_form_key=None,
):

    cluster_order = cluster_order or cluster_stats.keys()

    clusters = []

    for cluster_id in cluster_order:

        stats = cluster_stats[cluster_id]

        features = clustering[cluster_id]

        style = None
        if area_stat is not None:
            cluster_width = calculate_area_mark(stats[area_stat], total_doc_count)
            style = f"--w: {cluster_width:.3f}rem"

        display = None
        if display_stat is not None:
            display = h("span", klass="display-number")(stats[display_stat])

        see_more_link = None
        if seemore_url_key is not None:
            see_more_link = h(
                "a",
                href=stats[seemore_url_key],
            )(f"See all in cluster {cluster_id}")

        clusters.append(
            h("li")(
                h("div", klass="header stat-row")(
                    h("a", href=stats[header_url_key])(
                        h("h2")("Cluster: ", cluster_id),
                    ),
                    display,
                    h("div", klass="area-mark", style=style)(),
                ),
                render_feature_stats_as_dl(
                    features,
                    area_stat=area_stat,
                    display_stat=display_stat,
                    total_doc_count=total_doc_count,
                    feature_url_key=feature_url_key,
                    feature_form_id=feature_form_id,
                    feature_form_key=feature_form_key,
                ),
                h("div")(see_more_link),
            )
        )

    return h("ul", klass="cluster feature-clustering")(clusters)


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
        ("Home", "/"),
        ("Feature Lists", "/indexed-field/"),
        ("Browse", "/browse"),
        ("Drilldown", "/cluster/"),
    ]
    main_nav = generate_nav("Main", nav_links)

    sub_nav_links = sub_nav_links or {}
    sub_nav = [
        generate_nav(sub_nav_label, sub_nav_links)
        for sub_nav_label, sub_nav_links in sub_nav_links.items()
    ]

    extra_css = extra_css or ""

    column_flex = column_flex or {}

    return html(lang="en")(
        h("head")(
            h("meta", name="viewport", content="width=device-width, initial-scale=1"),
            h("title")(page_title),
            h("style")(raw(default_css), raw(extra_css)),
        ),
        h("body")(
            h("header")(main_nav, sub_nav),
            h("main")(
                h("div", klass="main-header")(body_header),
                h("div", klass="columns")(
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
