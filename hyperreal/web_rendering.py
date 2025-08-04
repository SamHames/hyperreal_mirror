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


ul li {
    list-style: none;
    margin-right: var(--s-1);
}

body {
    display: flex;
    flex-direction: column;
    max-height: 100vh;
}

main {
    display: flex;
    flex-direction: row;
    flex: 1;
    overflow: hidden;
    max-height: 100vh;
}

.column {
    overflow: scroll;
    margin: var(--s0);
    flex: 1;
    min-width: 30ch;
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

dl a {
    text-decoration: none;
    display: inline-block;
}

dd {
    margin-inline-start: var(--s-1);
}

dd > * {
    margin-inline-end: var(--s-1);
}

dt::after {
    content: ":";
}

.area-mark {
    height: calc(var(--w));
    width: calc(var(--w));
    display: inline-block;
    vertical-align: middle;
    background: black;
    margin-inline-end: var(--s-1);
    border-radius: 50%;
}

.feature-clustering {
    line-height: 130%;
}

.feature-clustering > * {
    margin-inline-end: var(--s1);
    margin-block-end: var(--s1);
    --space: var(--s-3);
    width: 10em;
    overflow-x: scroll;
    overflow-y: hidden;
}

.cluster-header {
    background: var(--header);
    padding: var(--s-2);
    text-align: right;
    border-bottom: solid;
    margin-block-end: var(--s-2);
}

.cluster-header h2 {
    display: inline-block;
    font-size: 100%;
    margin-inline-end: var(--s-1);
}

"""


def calculate_area_mark(normalised_area, total_doc_count, min_mapping_size=0.1):
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
    url_key=None,
    display_stat=None,
    area_stat=None,
    total_doc_count=None,
    klass="stack",
):

    feature_order = feature_order or feature_stats.keys()

    items = []
    last_field = None

    for feature in feature_order:
        field, html_value = feature
        details = feature_stats[feature]

        if field != last_field:
            items.append(h("dt")(h("em")(field)))

        style = False

        if area_stat is not None:
            area_side = calculate_area_mark(details[area_stat], total_doc_count)
            style = f"--w: {area_side:.3f}lh;"

        display = None
        if display_stat is not None:
            display = h("span")(details[display_stat])

        if url_key is not None:
            href = details[url_key]
            items.append(
                h("dd")(
                    h("a", style=style, href=href)(
                        h("div", klass="area-mark")(), html_value
                    ),
                    display,
                )
            )
        else:
            items.append(
                h("dd", style=style)(
                    h("div", klass="area-mark")(), h("span")(html_value), display
                )
            )

        last_field = field

    return h("dl", klass=klass)(items)


def render_feature_clustering(
    clustering,
    cluster_stats,
    total_doc_count,
    cluster_order=None,
    area_stat="relative_doc_count",
    url_key=None,
):

    cluster_order = cluster_order or cluster_stats.keys()

    clusters = []

    for cluster_id in cluster_order:

        stats = cluster_stats[cluster_id]

        features = clustering[cluster_id]

        cluster_width = calculate_area_mark(stats[area_stat], total_doc_count)
        style = f"--w: {cluster_width:.3f}lh"

        clusters.append(
            h("li")(
                h("div", klass="cluster-header")(
                    h("a", href=stats["url"])(
                        h("h2")(cluster_id),
                        h("div", klass="area-mark", style=style)(),
                    )
                ),
                render_feature_stats_as_dl(
                    features,
                    area_stat=area_stat,
                    total_doc_count=total_doc_count,
                    url_key=url_key,
                ),
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
    page_title, body_columns, sub_nav_links=None, sub_nav_label=None, extra_css=None
):
    """Render a complete page with navigation and a page title."""

    nav_links = [
        ("Home", "/"),
        ("Feature Lists", "/indexed-field/"),
        ("Browse", "/browse"),
    ]
    main_nav = generate_nav("Main", nav_links)

    sub_nav_links = sub_nav_links or {}
    sub_nav = [
        generate_nav(sub_nav_label, sub_nav_links)
        for sub_nav_label, sub_nav_links in sub_nav_links.items()
    ]

    extra_css = extra_css or ""

    return html(lang="en")(
        h("head")(
            h("meta", name="viewport", content="width=device-width, initial-scale=1"),
            h("title")(page_title),
            h("style")(raw(default_css), raw(extra_css)),
        ),
        h("body")(
            h("header")(main_nav, sub_nav),
            h("main")((h("div", klass="column")(col) for col in body_columns)),
        ),
    )


def list_docs(docs):

    # TODO: render selected documents via key and id through the web interface.
    # especially when we get to annotation.
    return h("ul", klass="stack")(h("li", klass="doc")(doc) for _, _, doc in docs)


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
