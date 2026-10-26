# Claude plugin namespace

The Claude plugin identity is `claude-obsidian`, so its commands are
`/claude-obsidian:*`. The repository, portable Python core, vault configuration
and transaction schemas use the same name. User vaults never need a rename or
content migration for a plugin identity change.

Version 3.0.0 alone shipped the plugin as `agricidaniel-obsidian`. Version 3.0.1
restores `claude-obsidian`, the name users have invoked since the first release.

`claude plugin validate` reports a plugin name starting with `claude-` as an
error, because the name reads as one of Anthropic's own plugins, and
`claude plugin init` and `claude plugin tag` refuse it. Claude Code still
installs and loads such a plugin. The owner kept the published name, so the
host validator's reserved-name error is an approved exception; the project's
own `package validate` accepts `claude-obsidian` and still rejects every other
reserved name. Anthropic's plugin directory would not list the plugin under
this name. See the
[current manifest reference](https://code.claude.com/docs/en/plugins-reference#name).

## Try it from a checkout

Load a local checkout without changing the installed plugin:

```bash
claude --plugin-dir /absolute/path/to/claude-obsidian
```

Invoke `/claude-obsidian:wiki`, `/claude-obsidian:save`, or the other skills
under the same prefix. If the same name is installed from the marketplace,
disable that copy for the session so the checkout is the one that loads.

## Published upgrade

The marketplace channel is `agricidaniel-claude-obsidian`. Its plugin entry is
`claude-obsidian`, matching the manifest, and the marketplace carries a
top-level `renames` map from `agricidaniel-obsidian` to `claude-obsidian`, as
the [documented rename mechanism](https://code.claude.com/docs/en/plugins/host-marketplace#rename-or-remove-a-plugin)
describes.

**From 2.x.** The plugin name never changed for you, so this is an ordinary
update:

```bash
claude plugin marketplace update agricidaniel-claude-obsidian
claude plugin update claude-obsidian@agricidaniel-claude-obsidian
```

Restart Claude Code. Commands stay `/claude-obsidian:*`.

**From 3.0.0.** Update the marketplace copy:

```bash
claude plugin marketplace update agricidaniel-claude-obsidian
```

Claude Code rewrites the `agricidaniel-obsidian@agricidaniel-claude-obsidian`
key in `enabledPlugins` and `pluginConfigs` back to `claude-obsidian` in the
user, project and local scopes, then uninstalls the 3.0.0 plugin. Install the
restored name once and restart:

```bash
claude plugin install claude-obsidian@agricidaniel-claude-obsidian
```

Managed settings cannot be rewritten by Claude Code. An administrator who
enabled `agricidaniel-obsidian@agricidaniel-claude-obsidian` there must change
the key back to `claude-obsidian@agricidaniel-claude-obsidian`.

The 3.0.0 map from `claude-obsidian` to `agricidaniel-obsidian` is removed,
because a name cannot be both current and former. `package validate` rejects a
renames chain that does not end in a listed plugin or `null`, and the test
suite requires the 3.0.0 identity to keep migrating to the current one.

### Verify

After the update or install, check that the expected version and source are
installed and enabled, and that no entry reports an error:

```bash
claude plugin list --json
```

In a disposable vault, confirm the namespace exposes `wiki`, `save`,
`wiki-ingest`, and `wiki-query` once; confirm explicit vault selection, a
reviewed Save/Ingest operation and a Query that leaves the vault unchanged. An
installed-plugin listing or metadata validator alone does not establish skill
loading or successful operation.

### Roll back

To return to the last 2.x release, replace the channel with one pinned to it:

```bash
claude plugin marketplace remove agricidaniel-claude-obsidian
claude plugin marketplace add AgriciDaniel/claude-obsidian#v2.2.0
claude plugin install claude-obsidian@agricidaniel-claude-obsidian
claude plugin list --json
```

Removing a marketplace also removes the plugins installed from it, so
reinstall from the pinned channel as shown. To move forward again, remove the
pinned channel, add `AgriciDaniel/claude-obsidian` without a ref, and install
`claude-obsidian@agricidaniel-claude-obsidian`. Vault files require no
rollback for an adapter-only change.

## Evidence and limits

With Claude Code 2.1.306 on 2026-10-21, an isolated `CLAUDE_CONFIG_DIR` and a
git marketplace served over local smart HTTP that moved between the v2.2.0,
3.0.0 and 3.0.1 trees:

- From 2.2.0, `marketplace update` then `plugin update` moved
  `claude-obsidian` from 2.2.0 to 3.0.1 with the enabled key unchanged.
- From 3.0.0, `marketplace update` rewrote the enabled key to
  `claude-obsidian@agricidaniel-claude-obsidian` and uninstalled
  `agricidaniel-obsidian`; one `plugin install` then installed 3.0.1, enabled.
- In a session loaded with `--plugin-dir`, all 15 skills and 3 agents were
  listed under `claude-obsidian:`.
- `claude plugin validate` reported one error, the reserved name, for the
  plugin and the marketplace.

Earlier 3.0.0 runs with Claude Code 2.1.302 and 2.1.304 established that a
marketplace without a `renames` entry leaves existing installs reporting
`Plugin not found in marketplace`, and that the pinned `v2.2.0` rollback
reinstalls files identical to the tag.

The GitHub-hosted 3.0.1 channel, project and local scopes, managed settings and
Windows were not exercised. Check the target host's help before use. Local
source changes do not authorize installation, commits, publishing, or account
changes. See the [installation and uninstall guide](install-guide.md).
