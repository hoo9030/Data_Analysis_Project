(() => {
  const apiBase = `${window.location.origin}/api`;
  let datasetCache = [];
  const schemaCache = new Map();

  function $(sel) { return document.querySelector(sel); }

  async function fetchJSON(url, opts) {
    const res = await fetch(url, opts);
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }
    if (!res.ok) throw new Error((data && (data.detail || data.message)) || res.statusText);
    return data;
  }

  function updateDatasetIdDatalist() {
    const dl = $('#dataset-ids');
    if (!dl) return;
    dl.innerHTML = '';
    datasetCache.forEach(it => {
      const opt = document.createElement('option');
      opt.value = it.id;
      opt.label = it.original_name || it.id;
      dl.appendChild(opt);
    });
  }

  async function refreshDatasetCache() {
    try {
      const data = await fetchJSON(`${apiBase}/datasets`);
      datasetCache = data.items || [];
      updateDatasetIdDatalist();
    } catch (_) { /* ignore */ }
  }

  async function updateColumnsDatalist(datasetId) {
    const dl = $('#columns-suggest');
    if (!dl) return;
    dl.innerHTML = '';
    if (!datasetId) return;
    const meta = datasetCache.find(x => x.id === datasetId);
    let columns = (meta && Array.isArray(meta.columns) && meta.columns) || null;
    if (!columns) {
      try {
        if (schemaCache.has(datasetId)) {
          columns = schemaCache.get(datasetId);
        } else {
          const info = await fetchJSON(`${apiBase}/datasets/${encodeURIComponent(datasetId)}/schema`);
          columns = info.columns || [];
          schemaCache.set(datasetId, columns);
        }
      } catch (_) {
        columns = [];
      }
    }
    (columns || []).forEach(c => {
      const opt = document.createElement('option');
      opt.value = c;
      dl.appendChild(opt);
    });
  }

  function attach() {
    // When dataset IDs change on forms that require a column, update column suggestions
    const idInputs = ['#dist-id', '#cast-source', '#fillna-source', '#drop-source', '#rename-source'];
    idInputs.forEach(sel => {
      const el = $(sel);
      if (el) el.addEventListener('change', (e) => updateColumnsDatalist(e.target.value.trim()));
    });
    // Also update on table click based on selected row
    const list = $('#datasets-list');
    if (list) list.addEventListener('click', (ev) => {
      const t = ev.target;
      if (!(t instanceof HTMLElement)) return;
      const tr = t.closest('tr');
      const id = tr && tr.getAttribute('data-id');
      if (id) updateColumnsDatalist(id);
    });
  }

  window.addEventListener('DOMContentLoaded', async () => {
    attach();
    await refreshDatasetCache();
    // Try seeding columns for the currently selected preview id after a brief delay
    setTimeout(() => {
      const pidEl = $('#preview-id');
      if (pidEl && pidEl.value) updateColumnsDatalist(pidEl.value.trim());
    }, 300);
  });
})();

