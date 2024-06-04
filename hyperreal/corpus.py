"""
A Hyperreal corpus describes how to access and display documents and features.

The corpus is the main site of customisation for the particulars of your
dataset/collection of documents. The interface is designed so that you never
need to hold large collections of data in memory at once by only handling
documents one at a time through generators.

"""

import abc
import logging
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from html import escape
from typing import Any, Protocol, Union
from xml.etree import ElementTree

from dateutil.parser import isoparse
from jinja2 import Template
from lxml.html import fragment_fromstring
from markupsafe import Markup

from hyperreal import utilities, value_handlers
from hyperreal.db_utilities import connect_sqlite, dict_factory

logger = logging.getLogger(__name__)

Feature = tuple[str, Hashable]
"""
A feature in a document is a field and a paired value.

Values must be Hashable, can't be None, and need to be storable in a single
SQLite column.

Examples of features could be:

- The datetime a document was created:
    `('created_at', datetime(2024, 1, 1))`
- A single word extracted from a text field:
    `('text', 'cat')`
- A numerical quantity, like the number of likes on a social media post:
    `('likes', 100)

"""

FieldValues = Union[list[Hashable], set[Hashable], Hashable]
DocFeatures = Mapping[str, FieldValues]
"""
An DocFeatures is the structured format that the Hyperreal index uses.

This indexable format of the document is a mapping of fields to values with
the following requirements:

- Field names must always be strings.
- Field values must be hashable, and not None.
- The contents of a field can be either a single value or a group of values.
- A single value by itself is just treated as a singular value in a document.
- A list of values indicates that this field has multiple values in a document
  and the values are order dependent - for example the words in a text field.
- A set of values indicates that this field has multiple values and order is
  *not* important - for example tags applied to a photograph.

An example DocFeatures could be:

```
{
    # A singular value field.
    'creation_date': date(2024, 03, 01)',

    # The tokens extracted from the text.
    'text': ['the', 'cat', 'sat', 'on', 'the', 'mat'],

    # The authors of the document.
    'authors': {'Jane', 'Arjun', 'Sally'}
}
```

"""

# pylint: disable=too-few-public-methods


class Corpus(Protocol):
    """
    A corpus describes how to identify and retrieve and display documents.

    Note that corpus objects must also be picklable.

    """

    field_values: Mapping[str, value_handlers.ValueHandler]
    """
    Maps fields to their ValueHandler that describes how to transform them.

    There must be an entry in field_values for every field output by the
    `doc_to_features` method.

    """

    CORPUS_TYPE: str
    """
    A name for this type of corpus.

    This will be used to make sure that the index and the corpus are consistent.

    """

    @abc.abstractmethod
    def docs(self, keys: Iterable) -> Iterable[tuple[Hashable, Any]]:
        """
        Retrieve documents identified by the given keys.

        The return must be an iterator (preferably a generator) of document
        keys and objects representing the document. There are no constraints
        on what the document *is* in Python.

        """

    @abc.abstractmethod
    def keys(self) -> Iterable:
        """
        Iterate through all keys in the corpus.

        This enables enumeration of the entire collection of documents and
        their identifiers.

        Keys may be any single value other than None that can be inserted into
        an SQLite table.

        """

    @abc.abstractmethod
    def doc_to_features(self, doc) -> DocFeatures:
        """Extract the features from the given document."""

    @abc.abstractmethod
    def doc_to_html(self, doc):
        """Return HTML representing the document."""

    def doc_to_str(self, doc) -> str:
        """Return a string representation of the document."""
        return str(doc)

    def close(self) -> None:
        """
        Close this corpus, cleaning up all open objects.

        There may be no cleanup necessary, in which case this is a no-op.

        """


class EmptyCorpus(Corpus):
    """
    A specialised corpus object that will not access or retrieve documents.

    This is useful only in places where you don't need to know anything about the
    documents or the extracted values, such as for running feature clustering
    algorithms over an index, which does not require access to the underlying
    documents.

    """

    field_values = defaultdict(value_handlers.NoopHandler)

    def docs(self, keys):
        return []

    def keys(self):
        return []

    def close(self):
        return

    def doc_to_features(self, doc):
        return {}

    def doc_to_html(self, doc):
        return ""

    def doc_to_str(self, doc):
        return ""


class SqliteBackedCorpus(Corpus):
    """A helper class for creating corpuses backed by SQLite databases."""

    # pylint: disable=abstract-method

    def __init__(self, db_path):
        """
        At a minimum you will need to pass in a path to a database.

        If you are implementing something based on this, you will need to
        either ensure that the db_path attribute is set in your `__init__`,
        or you can call `super().__init__(db_path)` inside your custom class
        definition.

        This handles some basic things like saving the database path and ensuring
        that the corpus object is picklable for multiprocessing.

        You will still need to define the docs and keys methods.

        """

        self.db_path = db_path
        self._db = None

    @property
    def db(self):
        """
        The connection to the SQLite database.

        This is initialised the first time it is needed.
        """

        if self._db is None:
            self._db = connect_sqlite(self.db_path, row_factory=dict_factory)

        return self._db

    def __getstate__(self):
        return self.db_path

    def __setstate__(self, db_path):
        self.__init__(db_path)

    def close(self):
        if self._db is not None:
            self._db.close()


class PlainTextSqliteCorpus(SqliteBackedCorpus):
    """
    A corpus for handling lines in a plaintext file as individual documents.

    This corpus identifies lines in the file according the processing order.

    """

    CORPUS_TYPE = "PlainTextSqliteCorpus"

    field_values = {"text": value_handlers.StringHandler()}

    def __init__(self, db_path, tokeniser=utilities.tokens):
        super().__init__(db_path)
        self.tokeniser = tokeniser

    def __getstate__(self):
        return self.db_path, self.tokeniser

    def __setstate__(self, state):
        self.__init__(state[0], tokeniser=state[1])

    def replace_docs(self, texts):
        """Replace the existing documents with texts."""
        self.db.execute("pragma journal_mode=WAL")
        self.db.execute("savepoint add_texts")

        try:
            self.db.execute("drop table if exists doc")
            self.db.execute(
                """
                create table doc(
                    doc_id integer primary key,
                    text not null
                )
                """
            )

            self.db.execute("delete from doc")
            self.db.executemany(
                "insert or ignore into doc(text) values(?)", ([t] for t in texts)
            )

        except Exception:
            self.db.execute("rollback to add_texts")
            raise

        finally:
            self.db.execute("release add_texts")

    def keys(self):
        return (r["doc_id"] for r in self.db.execute("select doc_id from doc"))

    def docs(self, keys):
        self.db.execute("savepoint docs")
        try:
            for key in keys:
                doc = list(
                    self.db.execute(
                        "select doc_id, text from doc where doc_id = ?", [key]
                    )
                )[0]
                yield key, doc

        finally:
            self.db.execute("release docs")

    def doc_to_features(self, doc):
        return {"text": self.tokeniser(doc["text"])}

    def doc_to_html(self, doc):
        return Markup(f'<p>{escape(doc["text"])}</p>)')

    def doc_to_str(self, doc):
        return doc["text"]


STACKEXCHANGE_HTML = """
<details>
    <summary>
        <em>{{ base_fields["QuestionTitle"] | e }}</em> -
        {{ base_fields["PostType"] }} from {{ base_fields["site_url"] }}
    </summary>

    <p>
        <small>
            <a href="{{ base_fields["LiveLink"] }}">Live Link</a>
            Copyright {{ base_fields["ContentLicense"]}} by
            {% if base_fields["OwnerUserId"] %}
            <a href="{{ '{}/users/{}'.format(base_fields["site_url"], base_fields["OwnerUserId"]) }}">
                {{ base_fields["DisplayName"] }}
            </a>
            {% else %}
            <Deleted User>
            {% endif %}
        </small>
    </p>

    {{ base_fields["Body"] }}

    <details>
        <summary>Tags:</summary>
        <ul>
            {% for tag in tags %}
            <li>{{ tag }}
            {% endfor %}
        </ul>
    </details>

    <details>
        <summary>Comments:</summary>
        <ul>
            {% for comment in user_comments %}
                <li>{{ comment["Text"] | e}}
                    <small>
                        Copyright {{ comment["ContentLicense"]}} by
                        {% if comment["UserId"] %}
                        <a href="{{ '{}/users/{}'.format(comment["site_url"], comment["UserId"]) }}">
                            {{ comment["DisplayName"] }}
                        </a>
                        {% else %}
                        <Deleted User>
                        {% endif %}
                    </small>
                </li>
            {% endfor %}
        </ul>
    </details>
</details>
"""


class StackExchangeCorpus(SqliteBackedCorpus):
    """
    A corpus for data dumped from StackExchange.

    All data from Stackexchange sites is available in a standard format at:

    https://archive.org/download/stackexchange/

    """

    CORPUS_TYPE = "StackExchangeCorpus"

    field_values = {
        "UserPosting": value_handlers.StringHandler(),
        "Post": value_handlers.StringHandler(),
        "CodeBlock": value_handlers.StringHandler(),
        "Tag": value_handlers.StringHandler(),
        "UserCommenting": value_handlers.StringHandler(),
        "Comment": value_handlers.StringHandler(),
        "Site": value_handlers.StringHandler(),
        "PostType": value_handlers.StringHandler(),
        "CreationDate": value_handlers.DateHandler(),
        "CreationMonth": value_handlers.DateHandler(),
        "CreationYear": value_handlers.DateHandler(),
    }

    def replace_sites_data(self, *archive_files):
        """
        Add the data from the given stackexchange archive files to this corpus.

        The input files are the .7z files directly downloaded from
        https://archive.org/download/stackexchange/, no additional processing
        is needed.

        To handle StackOverflow, you will need to pass the following
        files:

        - stackoverflow.com-Posts.7z
        - stackoverflow.com-Comments.7z
        - stackoverflow.com-Users.7z

        Note that you will also need ~200GiB of free disk space to handle
        stackoverflow itself - all other sites will be significantly
        smaller.

        If the site has already been added, all existing posts will be deleted
        and replaced with the content of the provided files.

        """

        self.db.execute("pragma journal_mode=WAL")
        self.db.executescript(
            """
            create table if not exists Site(
                site_id integer primary key,
                site_url unique
            );

            create table if not exists User(
                AboutMe text,
                CreationDate datetime,
                Location text,
                ProfileImageUrl text,
                WebsiteUrl text,
                AccountId integer,
                Reputation integer,
                Id integer,
                Views integer,
                UpVotes integer,
                DownVotes integer,
                DisplayName text,
                LastAccessDate datetime,
                site_id integer references Site,
                primary key (site_id, Id)
            );

            create table if not exists Post(
                -- doc_id is a surrogate key consisting of "site_id/Id"
                -- this is necessary so that we can track document keys
                -- consistently on rebuilding the index after updating site
                -- data
                doc_id primary key,
                site_id integer references Site,
                Id integer,
                OwnerUserId integer,
                AcceptedAnswerId integer references Post,
                ContentLicense text,
                ParentId integer references Post,
                Title text default '',
                FavoriteCount integer,
                Score integer,
                CreationDate datetime,
                ViewCount integer,
                Body text,
                LastActivityDate datetime,
                CommentCount integer,
                PostType text,
                unique(site_id, Id),
                foreign key (site_id, OwnerUserId) references User(site_id, Id)
            );

            create table if not exists comment(
                CreationDate datetime,
                ContentLicense text,
                Score integer,
                Text text,
                UserId integer,
                PostId integer,
                Id integer,
                site_id integer references Site,
                primary key (site_id, Id),
                foreign key (site_id, UserId) references User(site_id, Id),
                foreign key (site_id, PostId) references Post(site_id, Id)
            );

            create index if not exists post_comment on comment(site_id, PostId);
            create index if not exists site_post on Post(site_id, Id);
            create index if not exists site_user on User(site_id, Id);

            create table if not exists PostTag(
                site_id integer references Site,
                PostId integer,
                Tag text,
                primary key (site_id, PostId, Tag),
                foreign key (site_id, PostId) references Post
            );

            """
        )

        try:
            self.db.execute("begin")

            to_process = []

            # stackoverflow and only stackoverflow is split into multiple
            # files, so we need to handle it specially.
            stackoverflow_filenames = {
                "stackoverflow.com-Posts.7z": "Posts",
                "stackoverflow.com-Comments.7z": "Comments",
                "stackoverflow.com-Users.7z": "Users",
            }

            seen_stackoverflow = {}

            for file in set(archive_files):
                file_locations = {
                    "Posts": file,
                    "Comments": file,
                    "Users": file,
                }

                filename = os.path.basename(file)
                # Special case stackoverflow by checking all the files are present at
                # the end. Don't process this particular file if it is stackoverflow
                # data.
                if filename in stackoverflow_filenames:
                    seen_stackoverflow[stackoverflow_filenames[filename]] = file
                    continue

                url_candidate, extension = os.path.splitext(filename)
                if extension != ".7z":
                    raise ValueError(
                        f"{file} does not seem like a valid stackexchange archive file"
                    )

                to_process.append(("https://" + url_candidate, file_locations))

            if len(seen_stackoverflow) == 3:
                to_process.append(("https://stackoverflow.com", seen_stackoverflow))

            elif len(seen_stackoverflow) > 0:
                missing_files = [
                    file
                    for file in stackoverflow_filenames
                    if file not in seen_stackoverflow
                ]
                raise ValueError(
                    "Missing stackoverflow files - please add "
                    f"{missing_files} to process the full stackoverflow dataset."
                )

            for site_url, file_locations in to_process:
                logger.info("Processing data for %s", site_url)
                self._add_single_site(site_url, file_locations)

            self.db.execute("commit")

        except Exception:
            self.db.execute("rollback")
            raise

    def _add_single_site(self, site_url, file_locations):

        # pylint: disable=import-outside-toplevel
        try:
            from py7zr import SevenZipFile
        except ImportError as exc:
            raise ImportError(
                "The py7zr package needs to be installed for this functionality."
            ) from exc

        with tempfile.TemporaryDirectory() as temp_directory:
            # Process Posts, which includes both questions and answers.
            tag_splitter = re.compile(r"<|>|<>|\|")

            self.db.execute(
                "insert or ignore into Site(site_url) values(?)", [site_url]
            )
            site_id = list(
                self.db.execute(
                    "select site_id from Site where site_url = ?", [site_url]
                )
            )[0]["site_id"]

            self.db.execute("delete from Post where site_id = ?", [site_id])
            self.db.execute("delete from Comment where site_id = ?", [site_id])
            self.db.execute("delete from User where site_id = ?", [site_id])

            # Extract the posts file
            with SevenZipFile(file_locations["Posts"], "r") as archive_file:
                archive_file.extract(path=temp_directory, targets=["Posts.xml"])
                posts_file = os.path.join(temp_directory, "Posts.xml")

            tree = ElementTree.iterparse(posts_file, events=("end",))
            post_types = {"1": "Question", "2": "Answer"}

            for _, elem in tree:
                # We only consider questions and answers - SX uses other post types
                # to describe wiki's, tags, moderator nominations and more.
                if elem.attrib.get("PostTypeId") not in ("1", "2"):
                    elem.clear()
                    continue

                doc = defaultdict(lambda: None)
                doc.update(elem.attrib)
                doc["PostType"] = post_types[elem.attrib["PostTypeId"]]
                doc["site_id"] = site_id
                doc["doc_id"] = f"{site_id}/{doc['Id']}"

                self.db.execute(
                    """
                    replace into post values (
                        :doc_id,
                        :site_id,
                        :Id,
                        :OwnerUserId,
                        :AcceptedAnswerId,
                        :ContentLicense,
                        :ParentId,
                        :Title,
                        :FavoriteCount,
                        :Score,
                        :CreationDate,
                        :ViewCount,
                        :Body,
                        :LastActivityDate,
                        :CommentCount,
                        :PostType
                    )
                    """,
                    doc,
                )

                tag_insert = (
                    (site_id, elem.attrib["Id"], t)
                    for t in tag_splitter.split(elem.attrib.get("Tags", ""))
                    if t
                )

                self.db.executemany("replace into PostTag values(?, ?, ?)", tag_insert)

                # This is important when using iterparse to free memory from
                # processed nodes in the tree.
                elem.clear()

            os.remove(posts_file)

            with SevenZipFile(file_locations["Comments"], "r") as archive_file:
                archive_file.extract(path=temp_directory, targets=["Comments.xml"])
                comments_file = os.path.join(temp_directory, "Comments.xml")

            tree = ElementTree.iterparse(comments_file, events=("end",))

            for _, elem in tree:
                doc = defaultdict(lambda: None)
                doc.update(elem.attrib)
                doc["site_id"] = site_id

                self.db.execute(
                    """
                    replace into comment values (
                        :CreationDate,
                        :ContentLicense,
                        :Score,
                        :Text,
                        :UserId,
                        :PostId,
                        :Id,
                        :site_id
                    )
                    """,
                    doc,
                )
                elem.clear()

            os.remove(comments_file)

            with SevenZipFile(file_locations["Users"], "r") as archive_file:
                archive_file.extract(path=temp_directory, targets=["Users.xml"])
                users_file = os.path.join(temp_directory, "Users.xml")

            tree = ElementTree.iterparse(users_file, events=("end",))

            for _, elem in tree:
                doc = defaultdict(lambda: None)
                doc.update(elem.attrib)
                doc["site_id"] = site_id

                self.db.execute(
                    """
                    replace into user values (
                        :AboutMe,
                        :CreationDate,
                        :Location,
                        :ProfileImageUrl,
                        :WebsiteUrl,
                        :AccountId,
                        :Reputation,
                        :Id,
                        :Views,
                        :UpVotes,
                        :DownVotes,
                        :DisplayName,
                        :LastAccessDate,
                        :site_id
                    )
                    """,
                    doc,
                )
                elem.clear()

            os.remove(users_file)

    def docs(self, keys):
        self.db.execute("savepoint docs")

        try:
            for key in keys:
                doc = list(
                    self.db.execute(
                        """
                        SELECT
                            site_url,
                            site_id,
                            Id,
                            Title,
                            Body,
                            -- Used to retrieve both tags for the root
                            -- question, and the Question asked for answers -
                            -- as the these aren't present on answers, only
                            -- the root question.
                            coalesce(ParentId, Id) as TagPostId,
                            coalesce(
                                (
                                    select DisplayName
                                    from User
                                    where
                                        (User.site_id, User.Id) =
                                        (Post.site_id, Post.OwnerUserId)
                                ),
                                '<Deleted User>'
                            ) as DisplayName,
                            CreationDate,
                            PostType
                        from Post
                        inner join Site using(site_id)
                        where Post.doc_id = ?
                        """,
                        [key],
                    )
                )[0]

                doc["QuestionTitle"] = list(
                    self.db.execute(
                        """
                        select Title
                        from Post
                        where (site_id, Id) = (:site_id, :TagPostId)
                        """,
                        doc,
                    )
                )[0]["Title"]

                # Use the tags on the question, not the (absent) tags on the answer.
                doc["Tags"] = {
                    r["Tag"]
                    for r in self.db.execute(
                        """
                        select Tag
                        from PostTag
                        where (site_id, PostId) = (:site_id, :TagPostId)
                        """,
                        doc,
                    )
                }

                # Note we're indexing by AccountId, which is stable across all SX sites,
                # not the local user ID.
                doc["UserComments"] = list(
                    self.db.execute(
                        """
                        SELECT
                            coalesce(
                                (
                                    select DisplayName
                                    from User
                                    where
                                        (User.site_id, User.Id) =
                                        (comment.site_id, comment.UserId)
                                ),
                                '<Deleted User>'
                            ) as DisplayName,
                            Text
                        from comment
                        where (comment.site_id, comment.PostId) = (:site_id, :Id)
                        """,
                        doc,
                    )
                )

                yield key, doc

        finally:
            self.db.execute("release docs")

    def keys(self):
        return (r["doc_id"] for r in self.db.execute("select doc_id from Post"))

    TEMPLATE = Template(STACKEXCHANGE_HTML)

    def doc_to_html(self, doc):
        return Markup(
            self.TEMPLATE.render(
                base_fields=doc,
                tags=doc["Tags"],
                user_comments=doc["UserComments"],
            )
        )

    def doc_to_features(self, doc):
        """Prepare a document for indexing."""
        code_blocks = []

        if doc["Body"]:
            body_html = fragment_fromstring(doc["Body"], create_parent="div")

            for node in body_html.xpath("//pre/code"):
                # Extract text of code blocks
                code_blocks.append(" ".join(node.itertext()))
                # Then remove them, treat everything else as post text
                node.getparent().remove(node)

            # This is a little wrong, ideally we'd handle the cases where
            # the code blocks are removed with a sentinel...
            post_text = " ".join(body_html.itertext())

        else:
            post_text = ""

        creation_date = isoparse(doc["CreationDate"]).date()

        doc_features = {
            "UserPosting": doc["DisplayName"],
            "Post": utilities.tokens((doc["Title"] or ""))
            + utilities.tokens(post_text),
            "CodeBlock": [
                t for line in code_blocks for t in utilities.tokens(line) if line
            ],
            "Tag": doc["Tags"],
            # Comments from deleted users remain, but have no UserId associated.
            "UserCommenting": {u["DisplayName"] for u in doc["UserComments"]},
            # Note that all comments are treated as a single field - this is not ideal.
            "Comment": [
                t for c in doc["UserComments"] for t in utilities.tokens(c["Text"])
            ],
            "Site": doc["site_url"],
            "PostType": doc["PostType"],
            # Note rounded to the nearest UTC date to avoid too many values.
            "CreationDate": creation_date,
            "CreationMonth": creation_date.replace(day=1),
            "CreationYear": creation_date.replace(month=1, day=1),
        }

        return doc_features
