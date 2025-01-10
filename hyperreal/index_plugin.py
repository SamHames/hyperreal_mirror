"""
An index plugin extends the functionality of the index

This module provides the interface for plugins, enabling them to add new functionality
to the index. This includes a mechanism to manage additional state (on the index and
elsewhere) and migrate that state in response to changes.

The process of managing migrations and state for the plugin is also used to manage the
core index schema itself.

- how and why to write a plugin
- 'stateless' plugins
- 

(general link to an extension)

"""

import collections
from typing import Optional, Callable


class Migration:

    def __init__(
        self,
        from_version: Optional[str],
        to_version: str,
        steps: list[str | Callable],
    ):
        self.from_version = from_version
        self.to_version = to_version
        self.steps = steps
        self.validate_steps()

    def validate_steps(self):
        invalid_steps = [
            step for step in self.steps if not (isinstance(step, str) or callable(step))
        ]
        if invalid_steps:
            raise ValueError(
                "Steps must be either strs for SQL queries, or callable functions. "
                f"These steps are invalid: {invalid_steps}"
            )


class IndexPlugin:

    plugin_name: str
    migrations: list[Migration]
    current_version: str

    def __init__(
        self,
        plugin_name: str,
        current_version: Optional[str] = None,
        migrations: Optional[list[Migration]] = None,
    ):

        self.plugin_name = plugin_name
        self.current_version = current_version
        self.migrations = migrations or []

        self.setup_validate_migrations()

    def setup_validate_migrations(self):
        """
        Validate structure of migrations.

        A valid set of migrations needs to:

            - Either be empty, indicating no state is maintained (stateless plugin), or
              have a migration to the `current_version`.
            - Have only one migration per `from_version`. This is necessary to ensure
              that we always follow a consistent pathway through the migrations.

        """

        # An empty set of migrations is valid, and is treated as a stateless migration.
        if not self.migrations:
            self.version_map = {}
            return

        # Validate
        version_map = collections.defaultdict(list)
        for m in self.migrations:
            version_map[m.from_version].append(m)

        invalid_migrations = {
            version: migrations
            for version, migrations in version_map.items()
            if len(migrations) > 1
        }

        if invalid_migrations:
            raise ValueError(
                "The following migrations are invalid as they share the same "
                f"from_version: {invalid_migrations=}"
            )

        # Check that we can migrate to the current version. If no migrations are
        # specified we treat this as a stateless migration.
        to_versions = set(m.to_version for m in self.migrations)

        if self.current_version not in to_versions:
            raise ValueError(
                f"There is no migration to the current version {current_version=}."
            )

        self.version_migration_map = {v: m[0] for v, m in version_map.items()}
