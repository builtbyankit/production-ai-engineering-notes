(() => {
  'use strict';
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];
  const storage = { get: k => { try { return localStorage.getItem(k); } catch { return null; } }, set: (k,v) => { try { localStorage.setItem(k,v); } catch {} } };
  let completed;
  try { completed = new Set(JSON.parse(storage.get('pai-completed') || '[]').filter(n => Number.isInteger(n) && n > 0 && n <= 21)); } catch { completed = new Set(); }
  function progress() {
    $$('[data-progress-label]').forEach(n => n.textContent = `${completed.size} / 21`);
    $$('[data-progress-bar]').forEach(n => n.value = completed.size);
    $$('[data-nav-class]').forEach(n => $('.done-indicator',n).hidden = !completed.has(Number(n.dataset.navClass)));
    $$('[data-card-class]').forEach(n => $('.card-done',n).hidden = !completed.has(Number(n.dataset.cardClass)));
    $$('[data-mark-class]').forEach(n => { const done = completed.has(Number(n.dataset.markClass)); n.textContent = done ? '✓ Marked as read' : 'Mark as read'; n.classList.toggle('complete',done); n.setAttribute('aria-pressed',String(done)); });
  }
  $$('[data-mark-class]').forEach(b => b.addEventListener('click',() => { const n=Number(b.dataset.markClass); completed.has(n) ? completed.delete(n) : completed.add(n); storage.set('pai-completed',JSON.stringify([...completed])); progress(); }));
  progress();
  if (document.body.dataset.class) storage.set('pai-last-class', document.body.dataset.class);
  const last=Number(storage.get('pai-last-class'));
  if ($('[data-continue]') && last > 0 && last <= 21) { $('[data-continue]').href=`class-${String(last).padStart(2,'0')}.html`; $('[data-continue]').textContent=`Continue class ${String(last).padStart(2,'0')} →`; }

  function closeNav() { document.body.classList.remove('nav-open'); $('.nav-scrim').hidden=true; $('.nav-toggle').setAttribute('aria-expanded','false'); }
  $('.nav-toggle')?.addEventListener('click', () => { const open=!document.body.classList.contains('nav-open'); document.body.classList.toggle('nav-open',open); $('.nav-scrim').hidden=!open; $('.nav-toggle').setAttribute('aria-expanded',String(open)); });
  $('.nav-scrim')?.addEventListener('click',closeNav);
  document.addEventListener('keydown',e => { if (e.key==='Escape') closeNav(); });
  $$('[data-filter]').forEach(b => b.addEventListener('click', () => { $$('[data-filter]').forEach(n => { const active=n===b; n.classList.toggle('selected',active); n.setAttribute('aria-pressed',String(active)); }); $$('[data-group]').forEach(g => g.hidden=b.dataset.filter!=='all' && b.dataset.filter!==g.dataset.group); }));
  $$('.copy-code').forEach(b => b.addEventListener('click',async () => { try { await navigator.clipboard.writeText($('code',b.closest('.code-block')).textContent); b.textContent='Copied'; setTimeout(() => b.textContent='Copy',1800); } catch { b.textContent='Select to copy'; } }));

  const diagramSources=new Map($$('.mermaid').map(n=>[n,n.textContent]));
  window.paiRenderErrors=[];
  async function diagrams() {
    const api=window.mermaid?.default || window.mermaid;
    if (!diagramSources.size) return;
    if (!api) { window.paiRenderErrors.push('Diagram library unavailable'); return; }
    api.initialize({startOnLoad:false,securityLevel:'strict',theme:document.documentElement.dataset.theme==='dark'?'dark':'default',fontFamily:'Inter, system-ui, sans-serif',suppressErrorRendering:true,flowchart:{curve:'linear'}});
    for (const [node,source] of diagramSources) {
      node.removeAttribute('data-processed'); node.textContent=source;
      try { await api.run({nodes:[node]}); }
      catch(e) { window.paiRenderErrors.push(String(e)); node.textContent=source; node.closest('figure').querySelector('figcaption').textContent='Diagram source shown below'; }
    }
  }
  function math() {
    if (!window.katex) return;
    $$('[data-math]').forEach(n => { try { katex.render(n.dataset.math,n,{displayMode:n.classList.contains('math-block'),throwOnError:false,trust:false,strict:'warn'}); } catch(e) { window.paiRenderErrors.push(String(e)); } });
  }
  window.paiReady=(async()=>{math();await diagrams();return true;})();
  function themeLabel() { $('.theme-toggle')?.setAttribute('aria-label',document.documentElement.dataset.theme==='dark'?'Switch to light theme':'Switch to dark theme'); }
  themeLabel();
  $('.theme-toggle')?.addEventListener('click', () => { const next=document.documentElement.dataset.theme==='dark'?'light':'dark'; document.documentElement.dataset.theme=next; storage.set('pai-theme',next); themeLabel(); window.paiReady=diagrams(); });

  const dialog=$('.search-dialog'), input=$('#search-input'), results=$('.search-results'), status=$('.search-status');
  let searchPromise, readyIndex, debounce;
  const normalize=s=>s.normalize('NFKD').toLowerCase();
  function loadSearch() {
    if (!searchPromise) searchPromise=fetch('search-index.json').then(r=>{if(!r.ok)throw new Error('Search index unavailable');return r.json();}).then(data=>{readyIndex=data.map(c=>({...c,normalized:normalize(c.text),normalizedTitle:normalize(c.title),normalizedHeadings:normalize(c.headings)}));return readyIndex;});
    return searchPromise;
  }
  async function openSearch() { if (!dialog.open) dialog.showModal(); input.focus(); try { await loadSearch(); if(input.value.length>=2)runSearch(); } catch { status.textContent='Search could not load. Please try again when connected.'; } }
  $('.search-open')?.addEventListener('click',openSearch);
  $('.search-close')?.addEventListener('click',()=>dialog.close());
  dialog?.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
  document.addEventListener('keydown',e=>{const typing=/INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName)||document.activeElement?.isContentEditable;if((e.key==='/'&&!typing)||(e.key==='k'&&(e.ctrlKey||e.metaKey))){e.preventDefault();openSearch();}});
  function runSearch() {
    const query=normalize(input.value.trim()), words=query.split(/\s+/).filter(Boolean);
    results.replaceChildren();
    if(query.length<2){status.textContent='Enter at least two characters to search all notes.';return;}
    if(!readyIndex){status.textContent='Loading notes…';loadSearch().then(runSearch).catch(()=>status.textContent='Search could not load.');return;}
    const hits=readyIndex.filter(c=>words.every(w=>c.normalized.includes(w))).map(c=>({c,score:words.reduce((s,w)=>s+(c.normalizedTitle.includes(w)?20:0)+(c.normalizedHeadings.includes(w)?4:0),c.normalized.includes(query)?8:0)})).sort((a,b)=>b.score-a.score||a.c.number-b.c.number).slice(0,21);
    status.textContent=hits.length?`${hits.length} ${hits.length===1?'class':'classes'} found`:'No matching classes. Try a shorter or different term.';
    for(const {c} of hits){const a=document.createElement('a');a.className='search-result';a.href=c.url;const label=document.createElement('span');label.className='result-class';label.textContent=`CLASS ${String(c.number).padStart(2,'0')}`;const h=document.createElement('h3');h.textContent=c.title;const p=document.createElement('p');let at=c.normalized.indexOf(query);if(at<0)at=c.normalized.indexOf(words[0]);const start=Math.max(0,at-75);p.textContent=(start?'…':'')+c.text.slice(start,start+240)+(start+240<c.text.length?'…':'');a.append(label,h,p);results.append(a);}
  }
  input?.addEventListener('input',()=>{clearTimeout(debounce);debounce=setTimeout(runSearch,120);});
  if ('IntersectionObserver' in window) { const links=new Map($$('.page-toc nav a').map(a=>[decodeURIComponent(a.hash.slice(1)),a]));const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(entry.isIntersecting){links.forEach(a=>a.classList.remove('current'));links.get(entry.target.id)?.classList.add('current');}}},{rootMargin:'-100px 0px -65% 0px'});$$('.prose h2[id]').forEach(h=>observer.observe(h)); }
})();
