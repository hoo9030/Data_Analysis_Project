(() => {
  const apiBase = `${window.location.origin}/api`;

  function $(sel) { return document.querySelector(sel); }
  function el(tag, attrs = {}, ...children) {
    const e = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => {
      if (k === 'class') e.className = v; else if (k === 'html') e.innerHTML = v; else e.setAttribute(k, v);
    });
    children.flat().forEach(c => e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c));
    return e;
  }

  function renderTable(columns, rows) {
    const table = el('table', { class: 'table' });
    const thead = el('thead');
    const trh = el('tr');
    columns.forEach(c => trh.appendChild(el('th', {}, c)));
    thead.appendChild(trh);
    const tbody = el('tbody');
    rows.forEach(r => {
      const tr = el('tr');
      columns.forEach(c => tr.appendChild(el('td', {}, r[c] !== undefined ? String(r[c]) : '')));
      tbody.appendChild(tr);
    });
    table.appendChild(thead);
    table.appendChild(tbody);
    return table;
  }

  function renderMatrix(obj) {
    // obj: { rowName: { col: value } }
    const cols = new Set();
    Object.values(obj).forEach(r => Object.keys(r || {}).forEach(k => cols.add(k)));
    const columns = ['metric', ...Array.from(cols)];
    const rows = Object.entries(obj).map(([name, r]) => ({ metric: name, ...r }));
    return renderTable(columns, rows);
  }

  async function fetchJSON(url, opts) {
    const res = await fetch(url, opts);
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }
    if (!res.ok) throw new Error((data && (data.detail || data.message)) || res.statusText);
    return data;
  }

  async function loadInfo() {
    try {
      const info = await fetchJSON(`${apiBase}/info`);
      $('#app-info').textContent = `${info.name} v${info.version}`;
    } catch (e) {
      $('#app-info').textContent = '서버 정보 불러오기 실패';
    }
  }

  function renderDatasetList(items) {
    const table = el('table', { class: 'table' });
    const thead = el('thead');
    const trh = el('tr');
    ['id','original_name','size_bytes','created_at','actions'].forEach(c => trh.appendChild(el('th', {}, c)));
    thead.appendChild(trh);
    const tbody = el('tbody');
    (items || []).forEach(x => {
      const tr = el('tr', { 'data-id': x.id });
      tr.appendChild(el('td', {}, x.id));
      tr.appendChild(el('td', {}, x.original_name));
      tr.appendChild(el('td', {}, String(x.size_bytes)));
      tr.appendChild(el('td', {}, x.created_at));
      const actions = el('td');
      actions.append(
        el('button', { class: 'btn-action', 'data-act': 'preview', title: '미리보기' }, 'Preview'), ' ',
        el('button', { class: 'btn-action', 'data-act': 'describe', title: '요약' }, 'Describe'), ' ',
        el('a', { href: `${apiBase}/datasets/${encodeURIComponent(x.id)}/download`, target: '_blank' }, 'Download'), ' ',
        el('button', { class: 'btn-danger', 'data-act': 'delete', title: '삭제' }, 'Delete')
      );
      tr.appendChild(actions);
      tbody.appendChild(tr);
    });
    table.appendChild(thead);
    table.appendChild(tbody);
    return table;
  }

  async function refreshList() {
    const container = $('#datasets-list');
    container.innerHTML = '';
    try {
      const data = await fetchJSON(`${apiBase}/datasets`);
      const items = (data.items || []);
      container.appendChild(renderDatasetList(items));
      // Autofill last id to preview/describe inputs
      if (items.length) {
        const lastId = items[0].id;
        $('#preview-id').value = lastId;
        $('#describe-id').value = lastId;
      }
    } catch (e) {
      container.textContent = `목록 불러오기 실패: ${e.message}`;
    }
  }

  function bindUpload() {
    $('#upload-form').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const file = $('#csv-file').files[0];
      if (!file) return;
      const dsid = $('#dataset-id').value.trim();
      const form = new FormData();
      form.append('file', file);
      if (dsid) form.append('dataset_id', dsid);
      $('#upload-result').textContent = '업로드 중...';
      try {
        const res = await fetchJSON(`${apiBase}/datasets`, { method: 'POST', body: form });
        $('#upload-result').textContent = `업로드 완료: ${res.dataset_id}`;
        await refreshList();
      } catch (e) {
        $('#upload-result').textContent = `업로드 실패: ${e.message}`;
      }
    });
  }

  function bindPreview() {
    $('#preview-form').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const id = $('#preview-id').value.trim();
      const n = Number($('#preview-n').value || 20);
      const container = $('#preview-table');
      container.innerHTML = '';
      if (!id) { container.textContent = 'Dataset ID를 입력하세요'; return; }
      try {
        const data = await fetchJSON(`${apiBase}/datasets/${encodeURIComponent(id)}/preview?nrows=${n}`);
        container.appendChild(renderTable(data.columns, data.rows));
      } catch (e) {
        container.textContent = `미리보기 실패: ${e.message}`;
      }
    });
  }

  function bindDescribe() {
    $('#describe-form').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const id = $('#describe-id').value.trim();
      const limit = Number($('#describe-limit').value || 5000);
      const all = $('#describe-all').checked;
      const container = $('#describe-table');
      container.innerHTML = '';
      if (!id) { container.textContent = 'Dataset ID를 입력하세요'; return; }
      try {
        const data = await fetchJSON(`${apiBase}/datasets/${encodeURIComponent(id)}/describe?limit=${limit}&include_all=${all}`);
        container.appendChild(renderMatrix(data.stats));
      } catch (e) {
        container.textContent = `요약 실패: ${e.message}`;
      }
    });
  }

  function bindToolbar() {
    $('#refresh-list').addEventListener('click', refreshList);
    // Delegate actions on dataset rows
    $('#datasets-list').addEventListener('click', async (ev) => {
      const t = ev.target;
      if (!(t instanceof HTMLElement)) return;
      const act = t.getAttribute('data-act');
      if (!act) return;
      const tr = t.closest('tr');
      const id = tr?.getAttribute('data-id');
      if (!id) return;
      if (act === 'preview') {
        $('#preview-id').value = id;
        document.getElementById('preview-form').scrollIntoView({ behavior: 'smooth' });
      } else if (act === 'describe') {
        $('#describe-id').value = id;
        document.getElementById('describe-form').scrollIntoView({ behavior: 'smooth' });
      } else if (act === 'delete') {
        if (!confirm(`정말 삭제할까요?\n${id}`)) return;
        try {
          await fetchJSON(`${apiBase}/datasets/${encodeURIComponent(id)}`, { method: 'DELETE' });
          await refreshList();
        } catch (e) {
          alert(`삭제 실패: ${e.message}`);
        }
      }
    });
  }

  function bindNulls() {
    $('#nulls-form').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const id = $('#nulls-id').value.trim();
      const limitStr = $('#nulls-limit').value.trim();
      const limit = limitStr ? Number(limitStr) : null;
      const container = $('#nulls-table');
      container.innerHTML = '';
      if (!id) { container.textContent = 'Dataset ID를 입력하세요'; return; }
      const url = new URL(`${apiBase}/datasets/${encodeURIComponent(id)}/nulls`);
      if (limit) url.searchParams.set('limit', String(limit));
      try {
        const data = await fetchJSON(url.toString());
        const cols = ['column','total_rows','nulls','null_pct'];
        container.appendChild(renderTable(cols, data.items || []));
      } catch (e) {
        container.textContent = `결측치 조회 실패: ${e.message}`;
      }
    });
  }

  function bindCast() {
    $('#cast-form').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const source = $('#cast-source').value.trim();
      const column = $('#cast-column').value.trim();
      const to = $('#cast-type').value;
      const out = $('#cast-out').value.trim();
      const strict = $('#cast-strict').checked;
      const result = $('#cast-result');
      result.textContent = '변환 중...';
      if (!source || !column) { result.textContent = 'Source/Column을 입력하세요'; return; }
      try {
        const body = { column, to, mode: strict ? 'strict' : 'coerce' };
        if (out) body.out_id = out;
        const data = await fetchJSON(`${apiBase}/datasets/${encodeURIComponent(source)}/cast`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        result.textContent = `완료: ${data.dataset_id} (before_nulls=${data.before_nulls}, after_nulls=${data.after_nulls}, new_nulls=${data.coerced_new_nulls})`;
        // Refresh list to show the new dataset
        await refreshList();
        // Autofill ids for convenience
        $('#preview-id').value = data.dataset_id;
        $('#describe-id').value = data.dataset_id;
        $('#nulls-id').value = data.dataset_id;
        $('#cast-source').value = data.dataset_id;
      } catch (e) {
        result.textContent = `변환 실패: ${e.message}`;
      }
    });
  }

  window.addEventListener('DOMContentLoaded', async () => {
    bindUpload();
    bindPreview();
    bindDescribe();
    bindToolbar();
    bindNulls();
    bindCast();
    await loadInfo();
    await refreshList();
  });
})();
