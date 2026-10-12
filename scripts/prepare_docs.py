"""Prepare MkDocs input from the canonical, unchanged Markdown notes."""
from pathlib import Path
import json
import shutil
import re
from datetime import datetime
from build_collection import build_collection

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'mkdocs_docs'
build_collection(ROOT)
course = json.loads((ROOT / 'course.json').read_text())
count = len(course['classes'])
qa = sum(item['qa'] for item in course['classes'])
diagrams = sum(item['diagrams'] for item in course['classes'])
units = {'hr': 3600, 'min': 60, 's': 1}
seconds = sum(sum(int(n) * units[u] for n, u in re.findall(r'(\d+)\s*(hr|min|s)', item['duration'])) for item in course['classes'])
hours = seconds // 3600
first_date = datetime.strptime(course['classes'][0]['date'], '%Y%m%d').strftime('%-d %B %Y')
last_date = datetime.strptime(course['classes'][-1]['date'], '%Y%m%d').strftime('%-d %B %Y')

if DOCS.exists():
    shutil.rmtree(DOCS)
DOCS.mkdir()
for item in course['classes']:
    shutil.copy2(ROOT / 'notes' / item['file'], DOCS / item['file'])
revision = (ROOT / 'notes/Course_Quick_Revision.md').read_text()
(DOCS / 'Course_Quick_Revision.md').write_text(revision.replace('(Course_Notes_Index.md)', '(index.md)'))
shutil.copy2(ROOT / 'notes/Production_AI_Engineering_Notes.zip', DOCS / 'Production_AI_Engineering_Notes.zip')
shutil.copytree(ROOT / 'vendor', DOCS / 'assets/vendor')
shutil.copytree(ROOT / 'mkdocs_assets', DOCS, dirs_exist_ok=True)

lines = [
    '# Production AI Engineering — Class Notes',
    '',
    f'Notes for **{count} released classes** in the Krish Naik Academy course, covering approximately **{hours} hours** of teaching from **{first_date} to {last_date}**.',
    '',
    f'Each chapter includes explanations in teaching order, supplied code excerpts, lecture diagrams, live Q&A, key pointers, and a practice checklist. The collection contains **{qa} Q&A entries** and **{diagrams} diagrams**.',
    '',
    'Use the sidebar or the table below to open a class. Search the notes using the search box, or open the [quick revision guide](Course_Quick_Revision.md).',
    '',
    '| Class | Topic | Recording length |',
    '|---|---|---|',
]
for item in course['classes']:
    lines.append(f"| {item['number']:02} | [{item['title']}]({item['file']}) | {item['duration']} |")
lines += [
    '',
    '## Downloads and sources',
    '',
    '- [Download all notes](Production_AI_Engineering_Notes.zip).',
    '- [Download options](downloads.md), including the combined Markdown book.',
    f"- [Course dashboard]({course['courseUrl']}).",
    '',
    f'These independent study notes cover the released sessions in this snapshot. Instructor notebooks, diagrams, and technical references are linked in the corresponding chapters. Unresolved questions, failed demonstrations, and deferred topics remain identified. The broader future course roadmap extends beyond these {count} classes.',
    '',
]
(DOCS / 'index.md').write_text('\n'.join(lines))
(DOCS / 'downloads.md').write_text(f'''# Downloads

- [All notes in one ZIP](Production_AI_Engineering_Notes.zip): {count} class chapters, course index, revision guide, and combined Markdown book.
- [Combined Markdown book on GitHub](https://github.com/builtbyankit/production-ai-engineering-notes/blob/main/notes/Production_AI_Engineering_Complete_Notes.md).
- [Individual Markdown files on GitHub](https://github.com/builtbyankit/production-ai-engineering-notes/tree/main/notes).
- [Quick revision](Course_Quick_Revision.md).

The Markdown archive is useful for offline reading. Mermaid diagrams render in readers that support Mermaid; this MkDocs site renders them automatically.
''')
print(f"Prepared {len(course['classes'])} class notes and the revision guide for MkDocs.")
