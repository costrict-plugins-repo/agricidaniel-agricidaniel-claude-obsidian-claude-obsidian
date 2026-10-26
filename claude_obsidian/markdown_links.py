"""Shared, offset-preserving navigation parsing for the supported Markdown subset.

This is a navigation scanner, not a complete CommonMark or Obsidian renderer.
It handles comments, code, math, escapes, and inline path links. Reference-style
links, HTML anchors, and extension/plugin syntax are outside this subset.
"""
from __future__ import annotations

import bisect
import html
import re
from dataclasses import dataclass
from typing import Iterator

_FENCE = re.compile(r'(`{3,}|~{3,})(.*)$')
_LIST = re.compile(r'(?:[-+*]|\d{1,9}[.)])[ \t]+')
_QUOTE_MARKER = re.compile(r' {0,3}> ?')
_ATX_START = re.compile(r'#{1,6}(?:[ \t]|$)')
_THEMATIC_OR_UNDERLINE = re.compile(r'(?:(?:\*[ \t]*){3,}|(?:_[ \t]*){3,}|(?:-[ \t]*){3,}|=+[ \t]*|-+[ \t]*)$')
_THEMATIC_BREAK = re.compile(r'(?:(?:\*[ \t]*){3,}|(?:_[ \t]*){3,}|(?:-[ \t]*){3,})$')
_SETEXT_UNDERLINE = re.compile(r'(?:=+|-+)[ \t]*$')
_LIST_ITEM_START = re.compile(r'(?:[-+*]|\d{1,9}[.)])(?:[ \t]|$)')
_BACKTICK_RUN = re.compile(r'`+')
_INLINE_SPECIAL = re.compile(r'[\n\\<%`$]')
_NOT_NEWLINE = re.compile(r'[^\r\n]')
_LINE_BREAK = re.compile(r'\r\n?|\n')
_LABEL_TOKEN = re.compile(r'[\[\]\\`]')
# A line starting with any other non-space character is ordinary paragraph text.
_BLOCK_START_CHARS = frozenset('>-+*0123456789`~#<=_')
WIKILINK_RE = re.compile(r'(?P<embed>!)?\[\[(?P<body>[^\]\r\n]+?)\]\]')
_WIKILINK_STOP = re.compile(r'[\]\r\n]')
_PUNCTUATION_ESCAPE = re.compile(r'\\([!"#$%&\'()*+,\-./:;<=>?@\[\]\\^_`{|}~])')
_HTML_RAW_OPEN_RE = re.compile(r'^ {0,3}<(script|pre|style|textarea)(?:[ \t>]|$)', re.I)
_HTML_BLOCK_TAG_RE = re.compile(
    r'^ {0,3}</?(?:address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|'
    r'dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|'
    r'h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|nav|noframes|ol|'
    r'optgroup|option|p|param|search|section|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul)'
    r'(?:[ \t\r\n]|/?>|$)', re.I,
)


# A lone complete tag opens an HTML block only where no paragraph is open.
HTML_COMPLETE_TAG_RE = re.compile(
    r" {0,3}(?:<[A-Za-z][A-Za-z0-9-]*(?:[ \t]+[A-Za-z_:][A-Za-z0-9_.:-]*"
    r"(?:[ \t]*=[ \t]*(?:[^ \t\"'=<>`]+|'[^']*'|\"[^\"]*\"))?)*[ \t]*/?>"
    r"|</[A-Za-z][A-Za-z0-9-]*[ \t]*>)[ \t]*$"
)


def html_block_start(line: str) -> bool:
    """Report whether a line opens a raw-text or block-tag HTML block."""
    return bool(_HTML_RAW_OPEN_RE.match(line) or _HTML_BLOCK_TAG_RE.match(line))


def _html_block_end(text: str, start: int, line_end: int, *, complete_tag: bool = False) -> int | None:
    """Bounded raw HTML support for heading analysis, not HTML rendering."""
    line = text[start:line_end]
    if (opening := _HTML_RAW_OPEN_RE.match(line)) is not None:
        closer = re.compile(r'</' + opening.group(1) + r'[ \t]*>', re.I).search(text, start)
        if closer is None:
            return len(text)
        end = text.find('\n', closer.end())
        return len(text) if end < 0 else end + 1
    if _HTML_BLOCK_TAG_RE.match(line) or (complete_tag and HTML_COMPLETE_TAG_RE.match(line)):
        end = line_end
        while end < len(text):
            following = text.find('\n', end)
            following = len(text) if following < 0 else following + 1
            if not text[end:following].strip():
                break
            end = following
        return end
    return None


def iter_wikilinks(text: str) -> Iterator[re.Match[str]]:
    """Yield ``WIKILINK_RE.finditer`` matches without rescanning failed bodies.

    A body cannot contain ``]`` or a line break, so every opener before the
    first such terminator fails for the same reason as the first opener.
    """
    cursor = 0
    while (opening := text.find('[[', cursor)) >= 0:
        start = opening - 1 if opening > cursor and text[opening - 1] == '!' else opening
        match = WIKILINK_RE.match(text, start)
        if match is not None:
            yield match
            cursor = match.end()
            continue
        stop = _WIKILINK_STOP.search(text, opening + 2)
        if stop is None:
            return
        cursor = stop.start() if stop.start() > opening + 2 else opening + 1


def _quote_prefix(expanded: str, limit: int | None = None) -> tuple[int, int]:
    """Return blockquote depth and content column for a tab-expanded line."""
    depth = offset = 0
    while limit is None or depth < limit:
        marker = _QUOTE_MARKER.match(expanded, offset)
        if marker is None:
            break
        depth += 1
        offset = marker.end()
    return depth, offset


def _is_fence_opener(content: str) -> bool:
    fence = _FENCE.fullmatch(content)
    return fence is not None and (fence[1][0] != '`' or '`' not in fence[2])


def _inline_code_limit(text: str, index: int, baseline: int) -> int:
    """Return the end of the paragraph containing ``index``.

    CommonMark code spans are inline content, so a closer cannot be found
    across a blank line or a block that interrupts the paragraph. Lines
    indented past the container's code column remain paragraph continuation.
    """
    line_start = text.rfind('\n', 0, index) + 1
    line_end = text.find('\n', index)
    if line_end < 0:
        return len(text)
    first = text[line_start:line_end].rstrip('\r').expandtabs(4)
    depth, column = _quote_prefix(first)
    content = first[column:].lstrip(' ')
    if (item := _LIST.match(content)) is not None:
        content = content[item.end():]
    if _ATX_START.match(content):
        return line_end + 1
    cursor = line_end + 1
    while cursor < len(text):
        following_end = text.find('\n', cursor)
        following_end = len(text) if following_end < 0 else following_end + 1
        if text[cursor] not in _BLOCK_START_CHARS and not text[cursor].isspace():
            cursor = following_end  # Unindented prose continues the paragraph.
            continue
        line = text[cursor:following_end].rstrip('\r\n').expandtabs(4)
        line_depth, column = _quote_prefix(line)
        body = line[column:]
        content = body.lstrip(' ')
        if not content or line_depth > depth:
            return cursor
        indent = len(body) - len(content)
        # A Setext underline cannot be a lazy line, so it ends the paragraph
        # only inside the same container content.
        if (_SETEXT_UNDERLINE.match(content) and not _THEMATIC_BREAK.match(content)
                and (line_depth < depth or indent < baseline)):
            cursor = following_end
            continue
        if indent < baseline + 4 and (
            _is_fence_opener(content) or _ATX_START.match(content)
            or _THEMATIC_OR_UNDERLINE.match(content) or _LIST_ITEM_START.match(content)
            or content.startswith('<!--') or _HTML_RAW_OPEN_RE.match(content)
            or _HTML_BLOCK_TAG_RE.match(content)
        ):
            return cursor
        cursor = following_end
    return len(text)


def mask_protected_regions(
    text: str,
    frontmatter_end: int = 0,
    *,
    preserve_inline_code: bool = False,
    preserve_block_code: bool = False,
    preserve_math: bool = False,
    mask_html_blocks: bool = False,
) -> str:
    """Blank protected content without moving a source offset or newline.

    List continuation indentation is relative to the current list item's content
    column. A four-space nested list continuation can therefore contain real
    navigation while a code block indented four more spaces cannot.

    Heading labels may preserve inline code, including its delimiters. Their
    block structure must still come from the default mask, since multiline
    inline code can contain text that resembles a heading or underline.

    Heading-only HTML masking consumes recognized raw blocks before their
    literal Markdown can affect following blocks. Both options default off for
    existing navigation callers. Generic custom tags and full HTML rendering
    remain outside this scanner.
    """
    chars = list(text)
    # Destinations are literal URL/path text, not prose math or comments. The
    # outer scanner still masks a whole link when enclosed in code or comments.
    destinations = {link.destination_start: link.destination_end
                    for link in iter_markdown_links(text)}
    destinations.update({match.start(): match.end() for match in iter_wikilinks(text)})
    destination_starts = sorted(destinations)
    # Index maximal backtick runs once so each opener finds its closer by
    # bisection instead of rescanning the remaining document.
    runs_by_length: dict[int, list[int]] = {}
    for run in _BACKTICK_RUN.finditer(text):
        runs_by_length.setdefault(len(run[0]), []).append(run.start())
    code_limit = -1
    # A failed math closer search stays valid for later openers on its line.
    math_searches: dict[str, tuple[int, int, int]] = {}
    # Forward searches are cached so a long line is not rescanned per opener.
    line_limit = next_special = -1

    def blank(start: int, end: int) -> None:
        chars[start:end] = _NOT_NEWLINE.sub(' ', text[start:end])

    blank(0, min(frontmatter_end, len(text)))
    index = min(frontmatter_end, len(text))
    lists: list[tuple[int, int]] = []
    baseline = 0
    quote_depth = 0
    # Indented code cannot interrupt a paragraph, including lazy continuation.
    in_paragraph = False
    while index < len(text):
        if (index == 0 or text[index - 1] == '\n') and (
                text[index] in _BLOCK_START_CHARS or text[index].isspace()):
            end = text.find('\n', index)
            end = len(text) if end < 0 else end + 1
            line = text[index:end].rstrip('\r\n')
            expanded = line.expandtabs(4)
            depth, column = _quote_prefix(expanded)
            container = (quote_depth, lists, in_paragraph)
            if depth != quote_depth:
                # Lists are tracked relative to the current blockquote content.
                lists = []
                in_paragraph = in_paragraph and depth < quote_depth
                quote_depth = depth
            body = expanded[column:]
            indent = len(body) - len(body.lstrip(' '))
            if body.strip():
                while lists and indent < lists[-1][1]:
                    lists.pop()
                baseline = lists[-1][1] if lists else 0
                marker = _LIST.match(body, indent) if indent < baseline + 4 else None
                if marker:
                    lists.append((indent, marker.end()))
                    baseline = marker.end()
                fence_start = marker.end() if marker else indent
                content = body[fence_start:]
                if indent < baseline + 4 and _is_fence_opener(content):
                    fence = _FENCE.fullmatch(content)
                    assert fence is not None
                    marker_char, minimum = fence[1][0], len(fence[1])
                    close = re.compile(rf' {{0,{baseline + 3}}}{re.escape(marker_char)}{{{minimum},}}[ \t]*$')
                    fence_end = end
                    while fence_end < len(text):
                        next_end = text.find('\n', fence_end)
                        next_end = len(text) if next_end < 0 else next_end + 1
                        closing_line = text[fence_end:next_end].rstrip('\r\n').expandtabs(4)
                        closing_depth, closing_column = _quote_prefix(closing_line, quote_depth)
                        if closing_depth < quote_depth:
                            # A quoted fence ends with its blockquote.
                            break
                        closing_line = closing_line[closing_column:]
                        closing_indent = len(closing_line) - len(closing_line.lstrip(' '))
                        if closing_line.strip() and closing_indent < baseline:
                            # A list-contained fence ends with its container.
                            # Revisit this line as ordinary outside Markdown.
                            break
                        fence_end = next_end
                        if close.fullmatch(closing_line):
                            break
                    if not preserve_block_code:
                        blank(index, fence_end)
                    in_paragraph = False
                    index = fence_end
                    continue
                if not marker and indent >= baseline + 4 and not in_paragraph:
                    if not preserve_block_code:
                        blank(index, end)
                    index = end
                    continue
                if mask_html_blocks and (html_end := _html_block_end(
                        text, index, end, complete_tag=not in_paragraph)) is not None:
                    blank(index, html_end)
                    in_paragraph = False
                    index = html_end
                    continue
                stripped = content.lstrip(' ')
                in_paragraph = not (
                    (marker and not stripped) or _ATX_START.match(stripped)
                    or _THEMATIC_OR_UNDERLINE.match(stripped)
                    # Heading analysis treats HTML opening a container as a block.
                    or (mask_html_blocks and (html_block_start(stripped)
                                              or HTML_COMPLETE_TAG_RE.match(stripped)))
                )
                if in_paragraph and not marker and container[2] and depth < container[0]:
                    # A lazy continuation line keeps its blockquote open, so
                    # the next quoted line still continues the same paragraph.
                    quote_depth, lists = container[0], container[1]
            else:
                in_paragraph = False
        elif index == 0 or text[index - 1] == '\n':
            # Unindented prose closes lists and quotes and continues a paragraph,
            # except that a lazy line keeps an open quote paragraph's container.
            if not (quote_depth and in_paragraph):
                lists = []
                baseline = quote_depth = 0
            in_paragraph = True
        if index in destinations:
            index = destinations[index]
            continue
        if text[index] == '\\' and index + 1 < len(text):
            if text[index + 1] in '[!':
                blank(index + 1, index + 2)
            index += 2
            continue
        if text.startswith('<!--', index) or text.startswith('%%', index):
            delimiter, opening = ('-->', 4) if text.startswith('<!--', index) else ('%%', 2)
            closing = text.find(delimiter, index + opening)
            end = len(text) if closing < 0 else closing + len(delimiter)
            blank(index, end)
            index = end
            continue
        if text[index] == '`':
            end = index + 1
            while end < len(text) and text[end] == '`':
                end += 1
            length = end - index
            if index >= code_limit:
                code_limit = _inline_code_limit(text, index, baseline)
            starts = runs_by_length.get(length, [])
            position = bisect.bisect_left(starts, end)
            if position < len(starts) and starts[position] < code_limit:
                end = starts[position] + length
                if not preserve_inline_code:
                    blank(index, end)
            index = end
            continue
        if text[index] == '$':
            delimiter = '$$' if text.startswith('$$', index) else '$'
            search = index + len(delimiter)
            if delimiter == '$$':
                limit = len(text)
            else:
                if line_limit < search:
                    line_limit = text.find('\n', search)
                    line_limit = len(text) if line_limit < 0 else line_limit
                limit = line_limit
            searched_from, closing, searched_limit = math_searches.get(delimiter, (0, -1, -1))
            if not (searched_limit == limit and searched_from <= search
                    and (closing < 0 or search <= closing)):
                closing = text.find(delimiter, search, limit)
                while closing >= 0 and (text[closing - 1] == '\\' or (delimiter == '$' and text[closing - 1].isspace())):
                    closing = text.find(delimiter, closing + len(delimiter), limit)
                math_searches[delimiter] = (search, closing, limit)
            if closing > search and (delimiter == '$$' or not text[search].isspace()):
                end = closing + len(delimiter)
                if not preserve_math:
                    blank(index, end)
                index = end
                continue
        index += 1
        if index >= len(text) or text[index - 1] == '\n':
            continue
        # Ordinary prose cannot start a protected region; skip to the next
        # candidate delimiter, line start, or recognized destination.
        if next_special < index:
            special = _INLINE_SPECIAL.search(text, index)
            next_special = special.start() if special is not None else len(text)
        following = next_special
        position = bisect.bisect_left(destination_starts, index)
        if position < len(destination_starts):
            following = min(following, destination_starts[position])
        index = max(index, following)
    return ''.join(chars)


@dataclass(frozen=True)
class MarkdownLink:
    start: int
    end: int
    embed: bool
    destination: str
    destination_start: int
    destination_end: int
    label_start: int
    label_end: int


def markdown_destination(value: str) -> str:
    """Extract a destination, permitting angle delimiters and optional titles."""
    value = value.strip()
    if value.startswith('<'):
        closing = re.search(r'(?<!\\)>', value)
        if closing is None:
            return ''
        destination, suffix = value[1:closing.start()], value[closing.end():]
    else:
        match = re.match(r'(?:\\.|[^\s])+', value)
        if match is None:
            return ''
        destination, suffix = match[0], value[match.end():]
    if suffix and not re.fullmatch(r'\s+(?:"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\((?:\\.|[^)\\])*\))\s*', suffix):
        return ''
    return html.unescape(_PUNCTUATION_ESCAPE.sub(r'\1', destination))


def _balanced_label_ends(text: str) -> dict[int, int]:
    """Pair label brackets in one pass while ignoring escapes and code spans."""
    ends: dict[int, int] = {}
    breaks = _LINE_BREAK.finditer(text)
    offset = 0
    while offset < len(text):
        line_break = next(breaks, None)
        line_start = offset
        line_end = len(text) if line_break is None else line_break.start()
        line = text[offset:line_end]
        offset = len(text) if line_break is None else line_break.end()
        if '[' not in line:
            continue
        runs = []
        for run in _BACKTICK_RUN.finditer(line) if '`' in line else ():
            backslashes = 0
            before = run.start() - 1
            while before >= 0 and line[before] == '\\':
                backslashes += 1
                before -= 1
            if backslashes % 2 == 0:
                runs.append(run)
        nearest_by_length: dict[int, tuple[int, int]] = {}
        code_ends: dict[int, int] = {}
        for run in reversed(runs):
            length = len(run[0])
            if length in nearest_by_length:
                code_ends[run.start()] = nearest_by_length[length][1]
            nearest_by_length[length] = run.span()

        run_ends = {run.start(): run.end() for run in runs}
        stack: list[int] = []
        index = 0
        while (token := _LABEL_TOKEN.search(line, index)) is not None:
            index = token.start()
            if index in code_ends:
                index = code_ends[index]
                continue
            if index in run_ends:
                index = run_ends[index]
                continue
            char = line[index]
            if char == '\\':
                index += 2
                continue
            if char == '[':
                stack.append(line_start + index)
            elif char == ']' and stack:
                ends[stack.pop()] = line_start + index
            index += 1
    return ends


_DESTINATION_MODES = {'': 0, '<': 1, '"': 2, "'": 3}


def _destination_end(masked: str, start: int, memo: dict[int, int]) -> int:
    """Return the offset of a destination's closing parenthesis, or -1.

    After its first character the scan depends only on position and quoting
    mode, never on where it started: the character before ``start`` is the
    opening parenthesis, so a quote there cannot open a title. Each position
    following a parenthesis therefore memoizes the lowest relative depth its
    remaining line reaches, and a later opener inside an unclosed destination
    fails in constant time instead of rescanning the line.
    """
    index = start
    depth = 1
    mode = '<' if masked.startswith('<', start) else ''
    checkpoints: list[list[int]] = []
    while index < len(masked) and masked[index] not in '\r\n':
        if masked[index - 1] == '(':
            key = index * 4 + _DESTINATION_MODES[mode]
            known = memo.get(key)
            if known is not None and depth + known > 0:
                break
            checkpoints.append([key, depth, depth])
        char = masked[index]
        if char == '\\':
            index += 2
            continue
        if mode == '<':
            if char == '>':
                mode = ''
        elif mode:
            if char == mode:
                mode = ''
        elif char in '\"\'' and masked[index - 1].isspace():
            mode = char
        elif char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0:
                return index
            if depth < checkpoints[-1][2]:
                checkpoints[-1][2] = depth
        index += 1
    else:
        known = 0
    lowest = depth + known
    for key, at, low in reversed(checkpoints):
        lowest = min(lowest, low)
        memo[key] = lowest - at
    return -1


def iter_markdown_links(masked: str) -> Iterator[MarkdownLink]:
    """Yield inline Markdown links with balanced labels and destinations.

    A link label is then scanned for nested images, such as a badge inside a
    link. Image alt text is not navigation, so an image label is skipped.
    """
    opening = re.compile(r'(!?)\[')
    label_ends = _balanced_label_ends(masked)
    memo: dict[int, int] = {}
    cursor, limit = 0, len(masked)
    resume: list[tuple[int, int]] = []
    while True:
        match = opening.search(masked, cursor, limit)
        if match is None:
            if not resume:
                return
            cursor, limit = resume.pop()
            continue
        label_start = match.end()
        label_end = label_ends.get(label_start - 1)
        if label_end is None or label_end >= limit or not masked.startswith('(', label_end + 1):
            cursor = match.end()
            continue
        start = label_end + 2
        index = _destination_end(masked, start, memo)
        if index < 0 or index >= limit:
            cursor = match.end()
            continue
        destination = markdown_destination(masked[start:index])
        if destination:
            yield MarkdownLink(
                match.start(),
                index + 1,
                bool(match[1]),
                destination,
                start,
                index,
                label_start,
                label_end,
            )
        cursor = index + 1
        if not match[1]:
            resume.append((cursor, limit))
            cursor, limit = label_start, label_end


def markdown_visible_text(text: str) -> str:
    """Replace supported inline links with their visible source labels."""
    masked = mask_protected_regions(text)
    pieces: list[str] = []
    cursor = 0
    for link in iter_markdown_links(masked):
        if link.start < cursor:
            continue  # Nested inside a label that is already visible text.
        pieces.append(text[cursor:link.start])
        pieces.append(text[link.label_start:link.label_end])
        cursor = link.end
    pieces.append(text[cursor:])
    return ''.join(pieces)


def mask_comments(text: str) -> str:
    """Blank HTML and Obsidian comments while preserving code and math."""
    return mask_protected_regions(
        text,
        preserve_inline_code=True,
        preserve_block_code=True,
        preserve_math=True,
    )
