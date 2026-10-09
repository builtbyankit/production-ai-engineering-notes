(() => {
  async function render() {
    window.notesRenderErrors = [];
    document.querySelectorAll('.arithmatex').forEach(node => {
      const text = node.textContent.trim();
      const display = text.startsWith('\\[');
      if ((display && text.endsWith('\\]')) || (text.startsWith('\\(') && text.endsWith('\\)'))) {
        try { katex.render(text.slice(2,-2), node, {displayMode:display,throwOnError:false,trust:false}); }
        catch(error) { window.notesRenderErrors.push(String(error)); }
      }
    });
    const api = window.mermaid?.default || window.mermaid;
    if (!api) return;
    api.initialize({startOnLoad:false,securityLevel:'strict',theme:'default',fontFamily:'system-ui, sans-serif',suppressErrorRendering:true,flowchart:{curve:'linear'}});
    for (const node of document.querySelectorAll('.mermaid')) {
      const source = node.textContent;
      node.textContent = source;
      try { await api.run({nodes:[node]}); }
      catch(error) { node.textContent = source; window.notesRenderErrors.push(error.message || String(error)); }
    }
  }
  const start = () => window.notesReady = render();
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',start);
  else start();
})();
