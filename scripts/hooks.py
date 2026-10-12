"""Publish the generated static site directly, without a Jekyll pass."""
from pathlib import Path
import re
import json


def on_config(config, **kwargs):
    """Add new recorded classes to navigation from the collection manifest."""
    course = json.loads((Path(__file__).resolve().parents[1] / 'course.json').read_text())
    nav = config['nav']

    def listed_files(items):
        files = set()
        for item in items:
            for value in item.values():
                if isinstance(value, list):
                    files.update(listed_files(value))
                elif isinstance(value, str):
                    files.add(value)
        return files

    listed = listed_files(nav)
    for item in course['classes']:
        if item['file'] in listed:
            continue
        section = item.get('nav_section', 'New classes')
        group = next((entry[section] for entry in nav if section in entry and isinstance(entry[section], list)), None)
        if group is None:
            group = []
            index = next((i for i, entry in enumerate(nav) if 'Quick revision' in entry), len(nav))
            nav.insert(index, {section: group})
        label = item.get('nav_title', f"{item['number']:02} · {item['title']}")
        group.append({label: item['file']})
        listed.add(item['file'])
    config['site_description'] = f"Notes for {len(course['classes'])} released Krish Naik Academy classes, with code, diagrams, Q&A, and quick revision."
    return config


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
