"""Publish the generated static site directly, without a Jekyll pass."""
from pathlib import Path
import re


def on_page_markdown(markdown, **kwargs):
    """Keep headings and their sidebar labels plain, including future notes."""
    lines = []
    fence = None
    for line in markdown.splitlines(keepends=True):
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            kind = marker.group(1)[0]
            if fence is None:
                fence = kind
            elif fence == kind:
                fence = None
        elif fence is None:
            line = re.sub(r'^(#{1,6}\s+)[^\w\s]+\s+', r'\1', line)
        lines.append(line)
    return ''.join(lines)


def on_post_build(config, **kwargs):
    (Path(config['site_dir']) / '.nojekyll').touch()
