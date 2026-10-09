"""Prepare MkDocs input from the canonical, unchanged Markdown notes."""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'mkdocs_docs'
course = json.loads((ROOT / 'course.json').read_text())

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
    'Notes for **21 released classes** in the Krish Naik Academy course, covering approximately **80 hours** of teaching from **19 July to 4 October 2026**.',
    '',
    'Each chapter includes explanations in teaching order, supplied code excerpts, lecture diagrams, live Q&A, key pointers, and a practice checklist. The collection contains **932 Q&A entries** and **85 diagrams**.',
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
    'These independent study notes cover the released sessions in this snapshot. Instructor notebooks, diagrams, and technical references are linked in the corresponding chapters. Unresolved questions, failed demonstrations, and deferred topics remain identified. The broader future course roadmap extends beyond these 21 classes.',
    '',
]
(DOCS / 'index.md').write_text('\n'.join(lines))
(DOCS / 'downloads.md').write_text('''# Downloads

- [All notes in one ZIP](Production_AI_Engineering_Notes.zip): 21 class chapters, course index, revision guide, and combined Markdown book.
- [Combined Markdown book on GitHub](https://github.com/builtbyankit/production-ai-engineering-notes/blob/main/notes/Production_AI_Engineering_Complete_Notes.md).
- [Individual Markdown files on GitHub](https://github.com/builtbyankit/production-ai-engineering-notes/tree/main/notes).
- [Quick revision](Course_Quick_Revision.md).

The Markdown archive is useful for offline reading. Mermaid diagrams render in readers that support Mermaid; this MkDocs site renders them automatically.
''')
print(f"Prepared {len(course['classes'])} class notes and the revision guide for MkDocs.")
