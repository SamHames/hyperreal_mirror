"""
HTML generating functions for the web viewer.

This is primarily about the broader structure of generating complete HTML pages - what
is passed in to this is primarily self-rendering objects from other parts of Hyperreal.

"""

import math
from urllib.parse import quote

from tinyhtml import frag, h, html, raw

SI_ABBR = {0: "", 1: "k", 2: "M", 3: "G", -1: "m", -2: "μ", -3: "n"}


def format_si_magnitude(number):
    """Format a number as an SI magnitude with human readable suffix."""
    if number == 0:
        return "0"

    magnitude = math.log10(number)
    suffix_range = magnitude // 3

    precision_range = magnitude - (suffix_range * 3)

    if suffix_range == 0:
        approx = number
    else:
        approx = round(number / (10 ** (3 * suffix_range)), 1)

    return str(approx) + SI_ABBR[suffix_range]


def render_feature_group(
    feature_stats,
    footer=None,
    display_docs=False,
    display_hits=True,
    select_form_id=None,
    select_form_prefix="",
    highlight_features=None,
):
    highlight_features = highlight_features or set()

    header_rows = [
        h("th", klass="sr-only", scope="col")("Field"),
        h("th", scope="col")("Value"),
    ]

    if display_docs:
        header_rows.append(h("th", scope="col")("Docs"))

    if display_hits:
        header_rows.append(h("th", scope="col")("Hits"))
        header_rows.append(h("th", scope="col")("Sim."))

    if select_form_id:
        header_rows.append(h("th", scope="col")("Select"))

    column_style = f""

    header = h("thead")(h("tr")(header_rows))

    last_field = None

    rows = []

    last_field = None

    # Layout the table
    for i, (feature, stats) in enumerate(feature_stats.items()):
        row = []

        field = feature[0]

        field_css_class = "sr-only"
        if field != last_field:
            field_css_class = None

        last_field = field

        row.append(h("th", scope="row", klass=field_css_class)(field))

        html_values = feature[1:]

        if len(html_values) == 2:
            html_value = (html_values[0], "-", html_values[1])
        else:
            html_value = html_values[0]

        if "feature_url" in stats:
            display_value = h("a", href=stats["feature_url"])(html_value)

        row.append(h("th", scope="row")(display_value))

        if display_docs:
            row.append(h("td")(format_si_magnitude(stats["doc_count"])))

        select_id = f"feature-{select_form_prefix}-{i}"

        if display_hits:
            score = stats["jaccard_similarity"]
            style = f"--sim: {score:.3f};"
            display = f"{score:.3f}"

            if display[0] == "0":
                display = display[1:]

            row.append(
                h("td")(h("label", for_=select_id)(format_si_magnitude(stats["hits"])))
            )
            row.append(
                h("td", klass="intensity", style=style)(
                    h("span", klass="underline")(display)
                )
            )

        if select_form_id and stats.get("select_form_value"):
            row.append(
                h("td")(
                    h(
                        "input",
                        type="checkbox",
                        name="f",
                        form=select_form_id,
                        value=stats["select_form_value"],
                        id=select_id,
                    )
                )
            )
        row_class = "has-bar"

        if feature in highlight_features:
            row_class += " query-selected"

        rows.append(h("tr", klass=row_class)(row))

    if footer:
        rows.append(h("tfoot")(h("tr")(h("td", colspan=2)(footer))))

    return h("table", klass="feature-group")(header, rows)


def render_feature_clustering(
    feature_clustering,
    cluster_stats,
    display_docs=False,
    display_hits=True,
    select_form_id=None,
    footer=None,
    highlight_features=None,
    highlight_clusters=None,
):
    clusters = []

    highlight_features = highlight_features or set()
    highlight_clusters = highlight_clusters or set()

    for cluster_id, stats in cluster_stats.items():
        cluster_html_id = f"cluster-{cluster_id}"
        heading = h("h2", id=cluster_html_id)(
            h("a", href=stats["feature_url"])(f"Cluster {cluster_id}")
        )

        cluster_data = []
        if display_docs:
            cluster_data.append(
                h("div", klass="stack")(
                    h("dt")("Docs"), h("dd")(format_si_magnitude(stats["doc_count"]))
                )
            )

        select_id = f"select-{cluster_id}"

        if display_hits:
            cluster_data.append(
                h("div", klass="stack")(
                    h("dt")("Hits"), h("dd")(format_si_magnitude(stats["hits"]))
                )
            )

        header_class = "cluster group-header"
        style = None
        if display_hits:
            sim = f'{stats["jaccard_similarity"]:.3f}'
            style = f"--sim: {sim};"
            if sim[0] == "0":
                sim = sim[1:]

            header_class += " has-bar intensity"

            cluster_data.append(
                h("div", klass="stack")(
                    h("dt")("Sim."),
                    h("dd", klass="underline")(sim),
                )
            )

        if select_form_id is not None:
            cluster_data.append(
                h("div", klass="stack")(
                    h("dt")("Select"),
                    h("dd")(
                        h(
                            "input",
                            type="checkbox",
                            name="c",
                            form=select_form_id,
                            value=cluster_id,
                            id=select_id,
                            aria_labelled_by=cluster_html_id,
                        )
                    ),
                )
            )

        if cluster_id in highlight_clusters:
            header_class += " query-selected"

        header = h("div", klass=header_class, style=style)(
            heading, h("dl", klass="cluster")(cluster_data)
        )

        features = feature_clustering[cluster_id]

        group_footer = None
        if stats.get("expand_url", False):
            remaining_count = stats["matching_feature_count"] - len(features)
            group_footer = h(
                "a",
                href=stats["expand_url"],
            )("+ ", remaining_count, " more")

        feature_group = render_feature_group(
            features,
            display_hits=display_hits,
            display_docs=display_docs,
            select_form_id=select_form_id,
            select_form_prefix=cluster_id,
            footer=group_footer,
            highlight_features=highlight_features,
        )

        clusters.append(h("li", klass="stack")(header, feature_group))

    if footer:
        clusters.append(h("li")(footer))

    return h("ol", klass="stack feature-clustering")(clusters)


def render_feature_edit_forms(
    reverse_url,
    current_query,
):
    query_input = None
    if current_query is not None:
        query_input = h("input", type="hidden", name="query", value=current_query)

    operations = [
        ("Selected Features", [("new-cluster", "Create new cluster")]),
        (
            "Selected Clusters",
            [
                ("merge-clusters", "Merge selected clusters"),
                ("dissolve-clusters", "Dissolve selected clusters"),
                ("split-clusters", "Split selected clusters"),
                ("delete-clusters", "Delete selected clusters"),
                ("refine-clusters", "Refine selected clusters"),
            ],
        ),
        (
            "Select Clusters & Features",
            [
                ("add-to-search", "Add selected to current query"),
                ("new-search", "Start new query with selected"),
            ],
        ),
    ]

    operator_select = h("select", name="operation")(
        h("optgroup", label=label)(
            h("option", value=operation)(display) for operation, display in ops
        )
        for label, ops in operations
    )

    return h(
        "form",
        klass="cluster",
        id="edit-model-form",
        method="post",
        action=reverse_url("edit-model"),
    )(query_input, operator_select, h("button", type="submit")("Apply"))


def cluster_navigation(browse_url, cluster_ids, selected=None):
    """Generate a selector to jump to any cluster."""

    cluster_ids = sorted(cluster_ids)
    n_clusters = len(cluster_ids)

    if selected is not None:
        cluster_offset = cluster_ids.index(selected)
        next_cluster = cluster_ids[(cluster_offset + 1) % n_clusters]
        prev_cluster = cluster_ids[(cluster_offset - 1) % n_clusters]
        link_to = [
            ("First Cluster", cluster_ids[0]),
            ("Previous Cluster", prev_cluster),
            ("Next Cluster", next_cluster),
            ("Last Cluster", cluster_ids[-1]),
        ]
    else:
        link_to = [
            ("First Cluster", cluster_ids[0]),
            ("Last Cluster", cluster_ids[-1]),
        ]

    return h("ul", klass="cluster")(
        h("li")(
            h(
                "a",
                href=browse_url
                # Link to the cluster and expand to show all terms
                + f"?c={cluster_id}&expand={cluster_id}",
            )(text)
        )
        for text, cluster_id in link_to
    )


def generate_search(search_url, search_fields, current_query=None):
    q = None

    if current_query:
        q = (h("input", type="hidden", name="query", value=current_query)(),)

    all_ops = [
        (
            "OR",
            "disjunction",
        ),
        (
            "AND",
            "conjunction",
        ),
    ]
    operators = (
        h("select", name="operator", id="search-operators")(
            h("option", value=op)(disp) for disp, op in all_ops
        ),
    )

    return h("form", method="get", action=search_url, klass="cluster")(
        q,
        h("label", for_="search-field")("Search field:"),
        h("select", name="search-field", id="search-field")(
            h("option", value=field)(field) for field in search_fields
        ),
        h("label", for_="search-value")("for:"),
        h("input", type="text", name="search-value", id="search-value")(),
        h("label", for_="search-operators")("using operator:"),
        operators,
        h("button", type="submit")("Search"),
        h(
            "input",
            type="checkbox",
            name="add-to-current-query",
            id="add-to-current-query",
        )(),
        h("label", for_="add-to-current-query")("Add to currenty query"),
    )


def generate_nav(label, links, klass=None, search=None):
    """Generate a navigation element."""

    # Make sure this is a valid HTML id
    nav_label = "-".join(label.split())

    nav_id = f"nav-{nav_label}"
    label = h("span", klass="nav-label", id=nav_id)(label)

    items = [
        h("a", href=href)(link_text) if href else link_text for link_text, href in links
    ]

    if search:
        items.append(search)

    return h("nav", aria_labelled_by=nav_id, klass="cluster")(
        label, h("ul", klass="cluster")([h("li")(item) for item in items])
    )


def search_results_header(sample_doc_count, matching_doc_count):
    return f"{sample_doc_count} of {matching_doc_count} matching documents"


def full_page(
    reverse_url,
    page_title,
    body_columns,
    search_url=None,
    search_fields=None,
    search_current_query=None,
    body_header=None,
    sub_nav_links=None,
    sub_nav_label=None,
    extra_css=None,
):
    """Render a complete page with navigation and a page title."""

    search = None
    if search_url:
        search = generate_search(
            search_url, search_fields, current_query=search_current_query
        )

    nav_links = [
        ("Index Overview", reverse_url("home")),
        ("Browse", reverse_url("browse")),
    ]
    main_nav = generate_nav("Main", nav_links, search=search)

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
        h("body", klass="stack")(
            h("header", klass="bordered")(main_nav),
            h("main", klass="stack")(
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
    height: 100dvh;
    overflow: hidden;
}

main {
    display: flex;
    flex-direction: column;
    min-height: 0;
}

main > * {
    margin: 0 var(--s0);
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
    flex: 0 1 var(--column-width);
    width: auto;
}

.column > * {
    padding: 0 var(--s-2);
}

.column h1 {
    box-shadow: var(--s-3) var(--s-3) var(--s-3) 0 var(--border-color);
    background-color: black;
    color: white;
}

.scrollable {
    overflow-y: scroll;
    flex: 1;
    padding: var(--s-3);
}


h1 {
    font-size: 160%;
}

h2, h3 {
    font-size: 130%;
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

.feature-clustering {
    --space: var(--s1);
}

.feature-group {
    --space: var(--s-2);
    font-family: monospace, monospace;
    width: 100%;
    display: grid;
    grid-template-columns: 1fr repeat(4, auto);

    tr, thead, tbody, tfoot {
        display: grid;
        grid-template-columns: subgrid;
        grid-column: span 5;
    }

    th, td {
        min-width: 0;
    }

    a {
        text-decoration: none;
        display: block;
        overflow: clip;
        text-overflow: ellipsis;
    }
}


.feature-group tr:has(input:checked) {
    background-color: yellow;
}

.group-header {
    gap: var(--s0);
    padding: var(--s-3);
    width: 100%;
    font-family: monospace, monospace;
    border-bottom: var(--thin) solid var(--border-color);

    --bar-color: #00CED1;

    dl {
        --space: var(--s-2);
    }
    dt {
        font-weight: bold;
    }

    h2 {
        flex: 1;
    }
}

.query-selected {
    border: var(--s-3) solid red;
}

.group-header:has(input:checked) {
    background-color: yellow;
}

.feature-group thead {
    background-color: white;
    font-weight: bold;
}

.feature-group :is(tbody, tfooter) th {
    font-weight: normal;
}

.feature-group th {
    vertical-align: top;
    text-align: left;
}

.feature-group :is(th, td) {
    padding: var(--thin) var(--s-3);;
}

.feature-group tbody td {
    font-size: 90%;
    text-align: right;
}

.feature-group th:first-child {
    grid-column: span 5;
    background-color: white;
}

.feature-group tbody th:first-child:before {
    content: "Field: " / "";
    font-weight: bold;
}

.sr-only {
    clip: rect(1px, 1px, 1px, 1px);
    clip-path: inset(50%);
    height: 1px;
    width: 1px;
    margin: -1px;
    overflow: hidden;
    padding: 0;
    position: absolute; 
}

.feature-group, .group-header {
    input {
        margin: 0 auto;
        display: block;
    }
}

.feature-group label {
    display: block;
}

.feature-group thead th:nth-child(n+3) {
    text-align: right;
}

.has-bar {
    position: relative;
}

.intensity::after {
    content: '';
    position: absolute;
    left: 0;
    bottom: 0;
    height: 100%;
    background-color: rgb(from var(--bar-color, #FFD700) r g b / 40%);
    width: calc(var(--sim) * 100%);
    display: block;
    z-index: -1;
}

.underline {
    text-decoration: underline;
}

#edit-model-form {}

/*************/
"""
