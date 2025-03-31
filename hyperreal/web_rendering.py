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

dl {
    min-width: 10em;
}

dl a {
    text-decoration: none;
    min-width: 50%;
    display: inline-block;
}

dt {
    font-weight: bold;
}

dd {
    margin-left: var(--s-1);
}

.bar::before {
    height: 0.5em;
    width: var(--w);
    content: "";
    display: inline-block;
    margin-inline-end: var(--s-1);
    vertical-align: middle;
    background: repeating-linear-gradient(
        90deg,
        gray,
        gray 0.995em,
        white 0.01em,
        white 0.02em
    );
}


"""


def render_features_as_dl(
    feature_stats,
    feature_order=None,
    url_key=None,
    display_stat=None,
    bar_stat=None,
    bar_norm=None,
    klass=False,
):

    feature_order = feature_order or feature_stats.keys()

    items = []
    last_field = None

    for feature in feature_order:
        field, html_value = feature
        details = feature_stats[feature]

        if field != last_field:
            items.append(h("dt")(field))

        style = False

        if bar_stat is not None:
            bar_width = math.log10(details[bar_stat])

            # if bar_norm is not None:
            #     bar_width /= bar_norm

            # bar_width = max(bar_width**0.5 * 10, 0.1)

            style = f"--w: {bar_width:.2f}em;"

        display = None
        if display_stat is not None:
            display = details[display_stat]

        if url_key is not None:
            href = details[url_key]
            items.append(
                h("dd")(
                    h("a", klass="bar", style=style, href=href)(html_value), display
                )
            )
        else:
            items.append(h("dd", klass="bar", style=style)(html_value, display))

        last_field = field

    if klass:
        klass = f"stack {klass}"
    else:
        klass = "stack"
    return h("dl", klass=klass)(items)


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

    nav_links = [("Home", "/"), ("Feature Lists", "/indexed-field/")]
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


# TODO: should these be accumulated into a mixin for the web interface?
# IE - all of these methods requiring the index get the index from self?
def feature_overview_link(handler, feature):
    field, value = feature

    href = f"/indexed-field/{field}?v={handler.to_url(value)}"

    return h("a", href=href)(value)


def index_field_overview(indexed_fields):
    sub_nav_links = {
        "Indexed Fields": [(f, f"/indexed-field/{f}") for f in indexed_fields]
    }
    return full_page(
        f"Feature summary for field: {field}",
        [],
        sub_nav_links=sub_nav_links,
        sub_nav_label="Indexed Fields",
    )


def link_to_indexed_field_overview(field):
    return h("a", href=f"/indexed-field/{field}")(field)


def home_page(index_summary_table):
    """Render the home_page."""

    header = index_summary_table[0]
    data = index_summary_table[1:]

    # TODO: probably need an attribute for title and description on the corpus?
    table = h("table")(
        h("caption")("Overview of the indexed fields for this collection."),
        h("thead")(h("tr")(h("th", scope="col")(col_name) for col_name in header)),
        h("tbody")(
            (
                h("tr")(
                    h("th", scope="row")(link_to_indexed_field_overview(row[0])),
                    (h("td")(item) for item in row[1:]),
                )
                for row in data
            )
        ),
    )

    return full_page("Hyperreal Home", [table])
