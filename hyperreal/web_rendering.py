"""
HTML generating functions for the web viewer.

This is primarily about the broader structure of generating complete HTML pages - what
is passed in to this is primarily self-rendering objects from other parts of Hyperreal. 

"""

import math
from urllib.parse import quote

from tinyhtml import h, html, frag, raw


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
  --s-2: calc(var(--s-1) / var(--ratio));
  --s-1: calc(var(--s0) / var(--ratio));
  --s0: 1rem;
  --s1: calc(var(--s0) * var(--ratio));
  --s2: calc(var(--s1) * var(--ratio));
}

.cluster {
    display: flex;
    flex-wrap: wrap;
}

nav ul {
    list-style: none;
    margin-bottom: 0.5em;
}

header {
    padding: 0.5em;
    background-color: #efefef;
    height: fit-content;
}

ul li {
    margin-right: 0.5em;
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
    flex: 1;
    flex-basis: 40ch;
    margin: var(--s-1);
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

.area-mark {
    height: calc(var(--w));
    width: calc(var(--w));
    display: inline-block;
    vertical-align: middle;
    background: black;
    margin-inline-end: var(--s-1);
}


.feature-cluster {
}

.feature-cluster > * {
    margin-inline-end: var(--s1);
    margin-block-end: var(--s1);
    max-width: 10em;
    overflow: scroll;
}

"""


def calculate_root_scaling(total_doc_count, min_mapping_size=0.1):
    """
    Calculate a simple root factor for visualising marks.

    """
    return math.log10(min_mapping_size) / (-math.log10(total_doc_count))


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

    root_scale = 0.5
    if total_doc_count:
        root_scale = calculate_root_scaling(total_doc_count)

    for feature in feature_order:
        field, html_value = feature
        details = feature_stats[feature]

        if field != last_field:
            items.append(h("dt")(h("em")(field)))

        style = False

        if area_stat is not None:
            area_side = details[area_stat] ** root_scale
            style = f"--w: {area_side:.3f}lh;"

        display = None
        if display_stat is not None:
            display = details[display_stat]

        if url_key is not None:
            href = details[url_key]
            items.append(
                h("dd")(
                    h("a", style=style, href=href)(
                        h("div", klass="area-mark")(), html_value
                    ),
                    h("span")(display),
                )
            )
        else:
            items.append(
                h("dd", style=style)(
                    h("div", klass="area-mark")(),
                    html_value,
                    h("span")(display),
                )
            )

        last_field = field

    return h("dl", klass=klass)(items)


def render_feature_clustering(clustering, cluster_stats, total_doc_count):

    clusters = []

    for cluster_id, features in clustering.items():

        clusters.append(
            h("li")(
                h("h2")(cluster_id),
                render_feature_stats_as_dl(
                    features,
                    area_stat="relative_doc_count",
                    total_doc_count=total_doc_count,
                    klass="stack",
                ),
            )
        )

    return h("ul", klass="cluster feature-cluster")(clusters)


def generate_nav(label, links, klass=None):
    """Generate a navigation element."""

    nav_id = f"nav-{label}"
    label = h("span", id=nav_id)(label)
    return h("nav", aria_labelled_by=nav_id)(
        label,
        h("ul", klass="cluster")(
            h("li")(h("a", href=href)(link_text) if href else link_text)
            for link_text, href in links
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
