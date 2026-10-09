import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { marked, Renderer } from 'marked';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(root, 'docs');
const course = JSON.parse(await fs.readFile(path.join(root, 'course.json'), 'utf8'));
const classes = course.classes;
const groups = [
  { id: 'foundations', title: 'Onboarding & foundations', numbers: [1, 2, 3] },
  { id: 'transformers', title: 'Transformers', numbers: [4, 5, 6, 7, 8, 9, 10] },
  { id: 'tokenization', title: 'Tokenization', numbers: [11, 12, 13, 14] },
  { id: 'prompting', title: 'Prompting & structured output', numbers: [15, 16] },
  { id: 'dspy', title: 'DSPy', numbers: [17, 18] },
  { id: 'fine-tuning', title: 'Fine-tuning', numbers: [19] },
  { id: 'inference', title: 'Cache & attention variants', numbers: [20, 21] },
];
const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const slug = value => value.replace(/<[^>]*>/g, '').normalize('NFKD').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '-').replace(/^-|-$/g, '');
const filename = c => `class-${String(c.number).padStart(2, '0')}.html`;
const date = s => new Date(`${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}T12:00:00Z`).toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' });
const plain = text => text.replace(/```[\s\S]*?```/g, s => s.replace(/```[^\n]*/g, '')).replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1').replace(/\[([^\]]*)\]\([^)]*\)/g, '$1').replace(/<[^>]*>/g, '').replace(/[#*_`|]/g, '').replace(/\s+/g, ' ').trim();

await fs.mkdir(out, { recursive: true });
await fs.mkdir(path.join(out, 'assets'), { recursive: true });
await fs.cp(path.join(root, 'vendor'), path.join(out, 'assets', 'vendor'), { recursive: true });
await fs.cp(path.join(root, 'web'), path.join(out, 'assets'), { recursive: true });
await fs.cp(path.join(root, 'notes'), path.join(out, 'downloads'), { recursive: true });
await fs.writeFile(path.join(out, '.nojekyll'), '');

function renderMarkdown(source, revision = false) {
  const toc = [], seen = new Map();
  let math = false, diagrams = 0;
  // Protect TeX delimiters from Markdown's backslash escaping.
  const protectedSource = source.replace(/```[\s\S]*?```|\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)/g, (whole, block, inline) => {
    if (whole.startsWith('```')) return whole;
    math = true;
    const content = escape(block ?? inline);
    return block !== undefined ? `\n<div class="math-block" data-math="${content}">${content}</div>\n` : `<span class="math-inline" data-math="${content}">${content}</span>`;
  });
  const renderer = Object.assign(new Renderer(), {
    heading({ tokens, depth }) {
      const text = this.parser.parseInline(tokens);
      const base = slug(text) || 'section';
      const count = (seen.get(base) || 0) + 1; seen.set(base, count);
      const id = base + (count > 1 ? `-${count}` : '');
      if (depth === 2) toc.push({ id, text: text.replace(/<[^>]*>/g, '') });
      return `<h${depth} id="${id}">${text}<a class="heading-link" href="#${id}" aria-label="Link to this section">#</a></h${depth}>\n`;
    },
    code({ text, lang }) {
      if (lang?.trim() === 'mermaid') {
        diagrams++;
        return `<figure class="diagram"><pre class="mermaid">${escape(text)}</pre><figcaption>Lecture diagram ${diagrams}</figcaption><details class="diagram-source"><summary>Diagram source</summary><pre><code>${escape(text)}</code></pre></details></figure>\n`;
      }
      return `<div class="code-block"><div class="code-label">${escape(lang || 'text')}<button class="copy-code" type="button">Copy</button></div><pre><code class="language-${escape(lang || 'text')}">${escape(text)}</code></pre></div>\n`;
    },
    table(token) {
      let html = '<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable table"><table><thead><tr>';
      for (const cell of token.header) html += `<th>${this.parser.parseInline(cell.tokens)}</th>`;
      html += '</tr></thead><tbody>';
      for (const row of token.rows) {
        html += '<tr>';
        for (const cell of row) html += `<td>${this.parser.parseInline(cell.tokens)}</td>`;
        html += '</tr>';
      }
      return html + '</tbody></table></div>\n';
    },
    link({ href, title, tokens }) {
      const mdFile = href.split('#')[0];
      const targetClass = classes.find(c => c.file === mdFile);
      if (targetClass) href = filename(targetClass);
      else if (mdFile === 'Course_Notes_Index.md') href = 'index.html';
      else if (mdFile === 'Course_Quick_Revision.md') href = 'revision.html';
      else if (!/^(https?:|mailto:|#)/.test(href) && /\.(md|zip)$/.test(mdFile)) href = `downloads/${href}`;
      const external = /^https?:/.test(href) ? ' rel="noopener"' : '';
      return `<a href="${escape(href)}"${title ? ` title="${escape(title)}"` : ''}${external}>${this.parser.parseInline(tokens)}</a>`;
    },
  });
  const html = marked.parse(protectedSource, { renderer, gfm: true, breaks: false });
  return { html, toc, math, diagrams };
}

const bookIcon = '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 4.5h6a3 3 0 0 1 3 3V21a3 3 0 0 0-3-3H4V4.5Z"/><path d="M13 7.5a3 3 0 0 1 3-3h4V18h-4a3 3 0 0 0-3 3"/></svg>';
function sidebar(active) {
  return `<aside class="sidebar" id="course-nav"><a class="nav-home ${active === 'home' ? 'active' : ''}" href="index.html">Course overview</a><a class="nav-home ${active === 'revision' ? 'active' : ''}" href="revision.html">Quick revision</a><div class="reading-progress"><span>Your progress</span><strong data-progress-label>0 / 21</strong><progress data-progress-bar max="21" value="0" aria-label="Completed classes"></progress></div>${groups.map(g => `<div class="nav-group"><h2>${escape(g.title)}</h2>${g.numbers.map(n => { const c = classes.find(c => c.number === n); return `<a class="nav-class ${active === n ? 'active' : ''}" ${active === n ? 'aria-current="page"' : ''} href="${filename(c)}" data-nav-class="${n}"><span class="nav-number">${String(n).padStart(2, '0')}</span><span>${escape(c.title)}</span><span class="done-indicator" aria-label="Completed" hidden>✓</span></a>`; }).join('')}</div>`).join('')}<div class="sidebar-bottom"><a href="${course.courseUrl}" rel="noopener">Original course ↗</a><a href="${course.repositoryUrl}" rel="noopener">GitHub repository ↗</a></div></aside>`;
}

function shell({ title, active, body, toc = [], math = false, diagrams = 0, description }) {
  const tocHtml = toc.length ? `<aside class="page-toc"><p>ON THIS PAGE</p><nav aria-label="Page sections">${toc.map(t => `<a href="#${t.id}">${t.text}</a>`).join('')}</nav><a class="back-to-top" href="#top">Back to top ↑</a></aside>` : '';
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escape(title)} · Production AI Engineering Notes</title><meta name="description" content="${escape(description || 'Study notes for 21 Production AI Engineering classes, with code, diagrams, live Q&A and quick revision.')}"/><meta name="theme-color" content="#f7f8fb"><link rel="icon" href="assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="assets/styles.css">${math ? '<link rel="stylesheet" href="assets/vendor/katex/katex.min.css">' : ''}<script>try{document.documentElement.dataset.theme=localStorage.getItem('pai-theme')||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light')}catch{}</script></head>
<body id="top" ${Number.isInteger(active) ? `data-class="${active}"` : ''}><a class="skip-link" href="#main">Skip to content</a>
<header class="topbar"><button class="nav-toggle icon-button" aria-label="Open course navigation" aria-controls="course-nav" aria-expanded="false">☰</button><a href="index.html" class="brand"><span class="brand-icon">${bookIcon}</span><span>Production AI<span class="brand-sub">ENGINEERING NOTES</span></span></a><div class="top-actions"><button class="search-open" type="button"><span aria-hidden="true">⌕</span><span>Search all notes</span><kbd>/</kbd></button><button class="theme-toggle icon-button" type="button" aria-label="Toggle dark theme">◐</button><a class="github-link" href="${course.repositoryUrl}" rel="noopener">GitHub ↗</a></div></header>
${sidebar(active)}<div class="nav-scrim" hidden></div><main id="main" class="main ${toc.length ? 'with-toc' : ''}"><div class="main-column">${body}<footer class="site-footer"><p>Independent study notes from the Krish Naik Academy course.</p><p>Released sessions: 19 July–4 October 2026. Instructor sources and verification references are linked in each chapter.</p><div><a href="${course.courseUrl}">Course dashboard ↗</a><a href="${course.repositoryUrl}">Source repository ↗</a></div></footer></div>${tocHtml}</main>
<dialog class="search-dialog" aria-labelledby="search-title"><div class="search-heading"><h2 id="search-title">Search all 21 classes</h2><button class="search-close icon-button" aria-label="Close search">×</button></div><label class="sr-only" for="search-input">Search notes</label><input id="search-input" type="search" placeholder="Try KV cache, tokenization, or fine-tuning…" autocomplete="off"><p class="search-status" role="status">Search titles, explanations, code, and Q&A.</p><div class="search-results"></div></dialog>
${math ? '<script src="assets/vendor/katex/katex.min.js" defer></script>' : ''}${diagrams ? '<script src="assets/vendor/mermaid/mermaid.min.js" defer></script>' : ''}<script src="assets/site.js" defer></script></body></html>`;
}

const stats = { qa: classes.reduce((s, c) => s + c.qa, 0), diagrams: classes.reduce((s, c) => s + c.diagrams, 0) };
const filters = `<div class="catalog-filters" role="group" aria-label="Filter classes"><button data-filter="all" aria-pressed="true" class="selected">All classes</button>${groups.map(g => `<button data-filter="${g.id}" aria-pressed="false">${escape(g.title)}</button>`).join('')}</div>`;
const catalog = groups.map(g => `<section class="catalog-group" data-group="${g.id}"><div class="group-heading"><h2>${escape(g.title)}</h2><span>${g.numbers.length} ${g.numbers.length === 1 ? 'class' : 'classes'}</span></div><div class="class-grid">${g.numbers.map(n => { const c = classes.find(c => c.number === n); return `<a href="${filename(c)}" class="class-card" data-card-class="${n}"><div class="card-top"><span class="class-tag">CLASS ${String(n).padStart(2, '0')}</span><span class="card-done" hidden>✓ Read</span></div><h3>${escape(c.title)}</h3><p>${date(c.date)}</p><div class="card-bottom"><span>${c.duration.replace(/hr/g, 'h').replace(/min/g, 'm')}</span><span>${c.qa} Q&A <span aria-hidden="true">↗</span></span></div></a>`; }).join('')}</div></section>`).join('');
const homeBody = `<section class="hero"><p class="eyebrow">COURSE COMPANION · KRISH NAIK ACADEMY</p><h1>From fundamentals<br>to inference.</h1><p class="hero-description">Detailed notes for every released class in Production AI Engineering. Work through the explanations, trace the code, and revisit the questions that came up live.</p><div class="hero-actions"><a class="button primary" data-continue href="class-01.html">Start reading <span aria-hidden="true">→</span></a><a class="button secondary" href="downloads/Production_AI_Engineering_Notes.zip" download>Download all notes ↓</a></div><div class="stats"><div><strong>21</strong><span>class notes</span></div><div><strong>80h</strong><span>of teaching</span></div><div><strong>${stats.qa}</strong><span>Q&A entries</span></div><div><strong>${stats.diagrams}</strong><span>diagrams</span></div></div></section><section class="catalog"><div class="section-intro"><div><p class="eyebrow">YOUR STUDY PATH</p><h2>Explore the classes</h2></div><a href="revision.html">Open quick revision →</a></div>${filters}${catalog}</section><section class="download-panel"><div><p class="eyebrow">READ YOUR WAY</p><h2>Keep the whole course close.</h2><p>Use the revision guide for key pointers, or download the complete Markdown book and individual chapters.</p></div><div class="download-links"><a href="revision.html">Quick revision <span>→</span></a><a href="downloads/Production_AI_Engineering_Complete_Notes.md" download>Combined notes book <span>↓</span></a><a href="downloads/Production_AI_Engineering_Notes.zip" download>Complete notes package <span>↓</span></a></div></section>`;
await fs.writeFile(path.join(out, 'index.html'), shell({ title: 'Course overview', active: 'home', body: homeBody }));

const search = [];
let generatedDiagrams = 0;
for (const c of classes) {
  const source = await fs.readFile(path.join(root, 'notes', c.file), 'utf8');
  const bodySource = source.replace(/^# .+\n/, '').replace(/^### 📋 .+\n/m, '').replace(/^\*\*⏱️ Duration:\*\*.*\n/m, '');
  const rendered = renderMarkdown(bodySource);
  generatedDiagrams += rendered.diagrams;
  const prev = classes[c.number - 2], next = classes[c.number];
  const body = `<header class="chapter-header"><p class="eyebrow">CLASS ${String(c.number).padStart(2, '0')} <span>·</span> ${date(c.date)}</p><h1>${escape(c.title)}</h1><div class="chapter-meta"><span>${escape(c.duration)}</span><span>${c.qa} Q&A entries</span>${c.diagrams ? `<span>${c.diagrams} diagrams</span>` : ''}</div><div class="chapter-actions"><a href="${c.recording}" class="text-link">Class recording ↗</a><a href="downloads/${c.file}" class="text-link" download>Download Markdown ↓</a><button class="mark-complete" data-mark-class="${c.number}">Mark as read</button></div></header><article class="prose">${rendered.html}</article><nav class="chapter-pagination" aria-label="Adjacent classes">${prev ? `<a href="${filename(prev)}"><span>← PREVIOUS CLASS</span><strong>${escape(prev.title)}</strong></a>` : '<div></div>'}${next ? `<a href="${filename(next)}"><span>NEXT CLASS →</span><strong>${escape(next.title)}</strong></a>` : '<a href="revision.html"><span>KEEP REVISING →</span><strong>Quick revision guide</strong></a>'}</nav>`;
  await fs.writeFile(path.join(out, filename(c)), shell({ title: `Class ${c.number}: ${c.title}`, active: c.number, body, ...rendered, description: `Class ${c.number} notes on ${c.title}, with supplied code, diagrams, ${c.qa} live Q&A entries and practice actions.` }));
  search.push({ number: c.number, title: c.title, url: filename(c), headings: rendered.toc.map(t => t.text).join(' '), text: plain(source) });
}

const revisionSource = (await fs.readFile(path.join(root, 'notes', 'Course_Quick_Revision.md'), 'utf8')).replace(/^# .+\n/, '');
const revision = renderMarkdown(revisionSource, true);
await fs.writeFile(path.join(out, 'revision.html'), shell({ title: 'Quick revision', active: 'revision', body: `<header class="chapter-header"><p class="eyebrow">ALL CLASSES · KEY POINTERS</p><h1>Quick revision</h1><p class="hero-description">The core ideas from every class, together in one place. Open a chapter whenever you need the explanation behind a pointer.</p><a class="text-link" href="downloads/Course_Quick_Revision.md" download>Download revision notes ↓</a></header><article class="prose">${revision.html}</article>`, ...revision }));
await fs.writeFile(path.join(out, 'search-index.json'), JSON.stringify(search));
await fs.writeFile(path.join(out, '404.html'), shell({ title: 'Page not found', active: 'home', body: '<section class="hero"><p class="eyebrow">404</p><h1>Let’s get you back<br>to the notes.</h1><p class="hero-description">This page could not be found.</p><a class="button primary" href="index.html">Course overview →</a></section>' }).replace(/(href|src)="(?!https?:|#)([^"]+)"/g, `$1="${course.siteUrl}$2"`));
await fs.writeFile(path.join(out, 'sitemap.xml'), `<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${['index.html', 'revision.html', ...classes.map(filename)].map(f => `<url><loc>${course.siteUrl}${f}</loc></url>`).join('')}</urlset>`);
if (generatedDiagrams !== stats.diagrams) throw new Error(`Diagram count mismatch: ${generatedDiagrams} versus ${stats.diagrams}`);
console.log(`Built ${classes.length} class pages, ${generatedDiagrams} diagrams, course overview, revision and full-text search.`);
