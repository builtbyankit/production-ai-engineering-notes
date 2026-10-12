# Production AI Engineering — MkDocs Notes

Study notes for the 22 released Krish Naik Academy classes from 19 July 2026 to 10 October 2026, covering approximately 83 hours, 964 Q&A entries, and 89 diagrams.

The site uses MkDocs with its standard Read the Docs theme: a sidebar, search, readable class pages, syntax-highlighted code, equations, and Mermaid diagrams.

## Read and download

- [Hosted MkDocs site](https://builtbyankit.github.io/production-ai-engineering-notes/)
- [Markdown course index](notes/Course_Notes_Index.md)
- [Quick revision](notes/Course_Quick_Revision.md)
- [Combined Markdown book](notes/Production_AI_Engineering_Complete_Notes.md)
- [All notes in a ZIP](notes/Production_AI_Engineering_Notes.zip)
- [Original course](https://learn.krishnaikacademy.com/web/courses/details/6a16f8935e281281cd6b1128)

## Build and preview

Python 3.12 or newer is recommended.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_docs.py
mkdocs build --strict
mkdocs serve
```

The canonical Markdown notes live in `notes/`. Add each new class to `course.json`, with an optional `nav_section` and short `nav_title` for its sidebar entry. The preparation step refreshes the index, revision guide, combined book, ZIP, and totals, then copies the site inputs into the ignored `mkdocs_docs/` folder. MkDocs adds any new manifest classes to navigation and builds the site into `docs/`.

After changing notes or MkDocs configuration, rebuild and commit `docs/` with the source changes. GitHub Pages publishes from the `main` branch and `/docs` directory. The CI workflow checks the strict MkDocs build on pushes and pull requests.

## Sources

These are independent study notes. Full class transcripts and matching instructor resources supplied the course content. Each chapter credits its recording, notebooks, repositories, and verification references. Failed demonstrations, source differences, unresolved questions, and deferred topics remain explicit. The future course roadmap extends beyond this snapshot.

Instructor sources are credited to Krish Naik Academy and the relevant authors. Vendored Mermaid 11.12.0 and KaTeX 0.16.22 assets include their upstream licenses. Those licenses apply to the libraries themselves.
