# Production AI Engineering — Course Notes

**[Read the hosted notes](https://builtbyankit.github.io/production-ai-engineering-notes/)**

Study notes for all 21 released live classes in Krish Naik Academy’s Production AI Engineering course, from 19 July to 4 October 2026. The collection covers approximately 80 hours of teaching, with 932 Q&A entries and 85 lecture diagrams.

The site includes class navigation, full-text search, quick revision, rendered equations and Mermaid diagrams, code copying, dark mode, and reading progress saved in your browser.

## Read or download

- [Hosted course overview](https://builtbyankit.github.io/production-ai-engineering-notes/)
- [Quick revision](https://builtbyankit.github.io/production-ai-engineering-notes/revision.html)
- [Individual Markdown notes](notes/Course_Notes_Index.md)
- [Combined Markdown book](notes/Production_AI_Engineering_Complete_Notes.md)
- [Download all notes](notes/Production_AI_Engineering_Notes.zip)
- [Original course dashboard](https://learn.krishnaikacademy.com/web/courses/details/6a16f8935e281281cd6b1128)

The primary sources are the full class transcripts and matching instructor resources. Each chapter links its recording, notebooks, repositories, and verification references. Unresolved questions, failed demonstrations, saved-output differences, and deferred topics are identified. These are independent study notes; the wider course roadmap includes material beyond the 21 released sessions in this snapshot.

Instructor sources remain credited to Krish Naik Academy and the relevant authors. Source documents and notebooks are linked from the notes. The repository contains the generated notes, the static site, and its build files.

## Repository layout

- `notes/`: the 21 class notes, index, revision guide, combined book, and download archive.
- `docs/`: the generated GitHub Pages site, with local assets and downloadable notes.
- `web/`: site styling and browser interactions.
- `scripts/build.mjs`: the static site generator.
- `course.json`: chapter metadata.
- `vendor/`: pinned Mermaid and KaTeX assets with their upstream licenses.

## Rebuild

Node.js 20 or newer is required.

```sh
npm ci --ignore-scripts
npm run build
```

To preview locally, serve `docs/` with a static web server, for example:

```sh
python3 -m http.server 8000 --directory docs
```

GitHub Pages publishes the `docs/` directory on the `main` branch. After editing Markdown or site source, rebuild and commit the generated `docs/` files before pushing. No server, account, or API key is required to use the published site. Reading progress stays in local browser storage and is not uploaded.

The vendored libraries are Mermaid 11.12.0 and KaTeX 0.16.22. Markdown rendering uses the pinned Marked development dependency. Upstream library licenses are included in `vendor/`; those licenses apply to the libraries themselves.
