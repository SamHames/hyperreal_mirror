"""
HTML generating functions for the web viewer.

This is primarily about the broader structure of generating complete HTML pages - what
is passed in to this is primarily self-rendering objects from other parts of Hyperreal. 

"""

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
    background-color: lightcyan;
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
  margin-block-start: 2rem;
}

"""


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


def indexed_field_page(idx, indexed_fields, docs, field, features, order_by="value"):
    """Lists all of the features in a field."""

    handler = idx.field_handlers[field][0]

    # Cases to handle: range encoded fields and the default presentations of keys? range
    # features Not showing positions on fields for which it isn't relevant? Ie, based
    # on what is known about the field as indexed
    content = list_docs(docs), h("table")(
        h("caption")(f"Matching documents and positions for field {field}"),
        h("thead")(
            h("tr")(h("th", scope="col")(name) for name in ["Feature", "Documents"])
        ),
        (
            h("tr")(
                h("th", scope="row")(feature_overview_link(handler, feature)),
                h("td")(stats["doc_count"]),
            )
            for feature, stats in features.items()
        ),
    )

    sub_nav_links = {
        "Indexed Fields": [(f, f"/indexed-field/{f}") for f in indexed_fields]
    }
    return full_page(
        f"Feature summary for field: {field}",
        content,
        sub_nav_links=sub_nav_links,
        sub_nav_label="Indexed Fields",
    )


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
