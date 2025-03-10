"""
HTML generating functions for the web viewer.

This is primarily about the broader structure of generating complete HTML pages - what
is passed in to this is primarily self-rendering objects from other parts of Hyperreal. 

"""

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
    overflow: auto;
    flex: 1;
}

"""


def generate_nav(links, klass=None, label=None):
    """Generate a navigation element."""

    label = h("span", id="navlabel")(label) if label else None
    return h("nav")(
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
    main_nav = generate_nav(nav_links)

    sub_nav = (
        generate_nav(sub_nav_links, label=sub_nav_label) if sub_nav_links else None
    )

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


def indexed_field_page(field, features, indexed_fields, order_by="value"):
    """Lists all of the features in a field."""

    content = h("ol")(h("li")(f"{field}: {value}") for field, value in features.keys())

    sub_nav_links = [(f, f"/indexed-field/{f}") for f in indexed_fields]
    return full_page(
        f"Feature summary for field: {field}",
        [content],
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
