"""Rebuild the collection and downloads from the canonical class manifest."""
from pathlib import Path
from datetime import datetime
import json
import re
import zipfile


def build_collection(root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    notes = root / 'notes'
    manifest_path = root / 'course.json'
    course = json.loads(manifest_path.read_text())
    classes = course['classes']
    numbers = [c['number'] for c in classes]
    if numbers != list(range(1, len(classes) + 1)):
        raise ValueError('Class numbers must be consecutive and ordered')
    if len({c['recording'] for c in classes}) != len(classes):
        raise ValueError('Duplicate recording identity')
    if [c['date'] for c in classes] != sorted(c['date'] for c in classes):
        raise ValueError('Class dates must be chronological')

    texts = {}
    pointers = {}
    for item in classes:
        filename = item['file']
        if Path(filename).name != filename or not filename.endswith('_Notes.md'):
            raise ValueError(f'Unexpected class filename: {filename}')
        text = (notes / filename).read_text()
        if item['recording'] not in text:
            raise ValueError(f'Missing recording source: {filename}')
        for section in ('Quick Updates', "What's Next", 'Live Q&A Highlights', 'Key Pointers to Remember', 'Action Items After Class'):
            if section not in text:
                raise ValueError(f'Missing {section}: {filename}')
        if text.count('```') % 2:
            raise ValueError(f'Unbalanced code fences: {filename}')
        if re.search(r'\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{30,})', text):
            raise ValueError(f'Credential-shaped text in {filename}')
        if re.search(r'\bturn\d+(?:search|view)\d+\b|cite', text):
            raise ValueError(f'Unresolved tool citation in {filename}')
        qa = re.search(r'^## .*Live Q&A Highlights\s*\n(.*?)(?=^## |\Z)', text, re.M | re.S)
        rows = [line for line in qa.group(1).splitlines() if line.strip().startswith('|')] if qa else []
        if len(rows) < 3 or any(len(re.findall(r'(?<!\\)\|', row)) != 3 for row in rows):
            raise ValueError(f'Invalid Q&A table: {filename}')
        section = re.search(r'^## .*Key Pointers to Remember\s*\n(.*?)(?=^## |\Z)', text, re.M | re.S)
        if section is None:
            raise ValueError(f'Missing revision pointers: {filename}')
        pointers[item['number']] = re.sub(r'\n?---\s*$', '', section.group(1).strip()).strip()
        item.update(words=len(text.split()), qa=len(rows) - 2, diagrams=text.count('```mermaid'))
        texts[item['number']] = text.strip()
    manifest_path.write_text(json.dumps(course, indent=2, ensure_ascii=False) + '\n')

    count = len(classes)
    qa_total = sum(c['qa'] for c in classes)
    diagram_total = sum(c['diagrams'] for c in classes)
    units = {'hr': 3600, 'min': 60, 's': 1}
    seconds = sum(sum(int(n) * units[u] for n, u in re.findall(r'(\d+)\s*(hr|min|s)', c['duration'])) for c in classes)
    hours, minutes = seconds // 3600, seconds % 3600 // 60
    first = datetime.strptime(classes[0]['date'], '%Y%m%d').strftime('%-d %B %Y')
    last = datetime.strptime(classes[-1]['date'], '%Y%m%d').strftime('%-d %B %Y')
    intro = [
        '# Production AI / LLM Engineering — Course Notes', '',
        '**Provider:** Krish Naik Academy',
        f'**Released classes documented:** {count} · {first}–{last}',
        f'**Recording lengths:** approximately {hours} hours {minutes} minutes',
        f'**Collection:** {qa_total} live Q&A entries · {diagram_total} diagrams', '',
        f"[Course dashboard]({course['courseUrl']}) · [Instructor resource hub](https://krishnaikacademy.notion.site/Production-AI-Engineering-3a8eba9593d080d7b876e8ced7d541f1)", '',
        'Each chapter follows the class-notes format: explanations in teaching order, supplied examples, lecture diagrams, live Q&A, key pointers, upcoming topics, and a practice checklist. Full transcripts and substantive closing questions are the primary sources; matching instructor materials supply diagrams and verified code. Official documentation and research qualify technical claims where needed.', '',
        'This collection covers the available recordings documented through the latest session above. The induction describes a broader future roadmap. Failed demonstrations, saved-output differences, unresolved questions, and deferred topics remain identified.', '',
    ]
    index_path = notes / 'Course_Notes_Index.md'
    existing_index = index_path.read_text()
    appendix = ''
    if '## Preparation resources' in existing_index:
        appendix = '## Preparation resources' + existing_index.split('## Preparation resources', 1)[1]
        appendix = re.sub(r'alongside the \d+ live-class notes', f'alongside the {count} live-class notes', appendix)
    index = intro + ['## Classes', '', '| Class | Session | Notes | Recording length |', '|---|---|---|---|']
    for c in classes:
        date = datetime.strptime(c['date'], '%Y%m%d').strftime('%-d %b %Y')
        index.append(f"| {c['number']:02} | {date} | [{c['title']}]({c['file']}) | {c['duration']} |")
    index += ['', appendix.strip(), ''] if appendix else ['', '## Reading options', '', '- [Quick revision](Course_Quick_Revision.md)', '- [Combined book](Production_AI_Engineering_Complete_Notes.md)', '- [All notes in a ZIP](Production_AI_Engineering_Notes.zip)', '']
    index_path.write_text('\n'.join(index))

    revision = ['# Production AI / LLM Engineering — Quick Revision', '', f'Key pointers from all {count} documented classes. Open the linked chapter for explanations, code, diagrams, Q&A, and practice actions.', '', '[Course index](Course_Notes_Index.md)', '']
    for c in classes:
        revision += [f"## Class {c['number']:02}: {c['title']}", '', f"[Full class notes]({c['file']})", '', pointers[c['number']], '', '---', '']
    revision_path = notes / 'Course_Quick_Revision.md'
    revision_path.write_text('\n'.join(revision))

    book = intro + ['## Classes in this book', '']
    for c in classes:
        book.append(f"- [Class {c['number']:02}: {c['title']}](#class-{c['number']:02})")
    book += ['', '---', '']
    for c in classes:
        book += [f'<a id="class-{c["number"]:02}"></a>', '', texts[c['number']], '', '[Back to course index](Course_Notes_Index.md)', '', '---', '']
    book_path = notes / 'Production_AI_Engineering_Complete_Notes.md'
    book_path.write_text('\n'.join(book))

    latest = datetime.strptime(classes[-1]['date'], '%Y%m%d')
    archive_path = notes / 'Production_AI_Engineering_Notes.zip'
    archive_files = [index_path, revision_path, book_path] + [notes / c['file'] for c in classes]
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for file in archive_files:
            info = zipfile.ZipInfo(file.name, (latest.year, latest.month, latest.day, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, file.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        if archive.testzip():
            raise ValueError('Archive integrity check failed')

    readme = root / 'README.md'
    summary = f'Study notes for the {count} released Krish Naik Academy classes from {first} to {last}, covering approximately {hours} hours, {qa_total} Q&A entries, and {diagram_total} diagrams.'
    readme.write_text(re.sub(r'^Study notes for the .+$', summary, readme.read_text(), flags=re.M))
    return {'classes': count, 'qa': qa_total, 'diagrams': diagram_total, 'hours': hours, 'minutes': minutes, 'archive_files': len(archive_files)}


if __name__ == '__main__':
    print(json.dumps(build_collection(), indent=2))
