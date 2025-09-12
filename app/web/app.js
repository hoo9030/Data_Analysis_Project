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

  function renderBars(items, labelKey, countKey) {
    const max = items.reduce((m, x) => Math.max(m, Number(x[countKey] || 0)), 0) || 1;
    const table = el('table', { class: 'table' });
    const thead = el('thead');
    thead.appendChild(el('tr', {}, el('th', {}, 'label'), el('th', {}, 'count'), el('th', {}, 'chart')));
    table.appendChild(thead);
    const tbody = el('tbody');
    items.forEach(it => {
      const label = String(it[labelKey] ?? '');
      const cnt = Number(it[countKey] || 0);
      const w = Math.max(2, Math.round(cnt / max * 100));
      const bar = el('div', { class: 'bar', style: `width:${w}%` });
      const tr = el('tr', {}, el('td', {}, label), el('td', {}, String(cnt)), el('td', {}, bar));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
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

  function bindDistribution() {
    $('#dist-form').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const id = $('#dist-id').value.trim();
      const col = $('#dist-col').value.trim();
      const bins = Number($('#dist-bins').value || 20);
      const topk = Number($('#dist-topk').value || 20);
      const dropna = $('#dist-dropna').checked;
      const container = $('#dist-view');
      container.innerHTML = '';
      if (!id || !col) { container.textContent = 'Dataset/Column을 입력하세요'; return; }
      const url = new URL(`${apiBase}/datasets/${encodeURIComponent(id)}/distribution`);
      url.searchParams.set('column', col);
      url.searchParams.set('bins', String(bins));
      url.searchParams.set('topk', String(topk));
      url.searchParams.set('dropna', String(dropna));
      try {
        const data = await fetchJSON(url.toString());
        if (data.type === 'numeric') {
          container.appendChild(el('div', { class: 'muted' }, `min=${data.min}, max=${data.max}, bins=${data.bins}, total=${data.total}, na=${data.na_count}`));
          const items = (data.items || []).map(x => ({ label: x.label || `${x.left}~${x.right}`, count: x.count }));
          container.appendChild(renderBars(items, 'label', 'count'));
        } else {
          container.appendChild(el('div', { class: 'muted' }, `topk=${data.topk}, total=${data.total}, na=${data.na_count}, unique=${data.unique}`));
          container.appendChild(renderBars(data.items || [], 'value', 'count'));
        }
      } catch (e) {
        container.textContent = `분포 계산 실패: ${e.message}`;
      }
    });
  }

  function bindCorrelation() {
    $('#corr-form').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const id = $('#corr-id').value.trim();
      const method = $('#corr-method').value;
      const limit = Number($('#corr-limit').value || 50000);
      const container = $('#corr-view');
      container.innerHTML = '';
      if (!id) { container.textContent = 'Dataset ID를 입력하세요'; return; }
      const url = new URL(`${apiBase}/datasets/${encodeURIComponent(id)}/corr`);
      url.searchParams.set('method', method);
      if (limit) url.searchParams.set('limit', String(limit));
      try {
        const data = await fetchJSON(url.toString());
        container.appendChild(renderMatrix(data.matrix || {}));
      } catch (e) {
        container.textContent = `상관분석 실패: ${e.message}`;
      }
    });
  }

  function bindSampleFilter() {
    $('#sample-form').addEventListener('submit', (ev) => {
      ev.preventDefault();
      const id = $('#sample-id').value.trim();
      const rows = Number($('#sample-rows').value || 100);
      if (!id) return;
      const url = new URL(`${apiBase}/datasets/${encodeURIComponent(id)}/sample.csv`);
      url.searchParams.set('rows', String(rows));
      window.open(url.toString(), '_blank');
    });

    $('#filter-form').addEventListener('submit', (ev) => {
      ev.preventDefault();
      const id = $('#filter-id').value.trim();
      const cols = $('#filter-cols').value.trim();
      const limit = Number($('#filter-limit').value || 10000);
      const query = $('#filter-query').value.trim();
      if (!id) return;
      const url = new URL(`${apiBase}/datasets/${encodeURIComponent(id)}/filter.csv`);
      if (cols) url.searchParams.set('columns', cols);
      if (limit >= 0) url.searchParams.set('limit', String(limit));
      if (query) url.searchParams.set('query', query);
      window.open(url.toString(), '_blank');
    });
  }

  function bindPreprocess() {
    // fillna
    $('#fillna-form').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const source = $('#fillna-source').value.trim();
      const cols = $('#fillna-cols').value.trim();
      const strategy = $('#fillna-strategy').value;
      const value = $('#fillna-value').value;
      const out = $('#fillna-out').value.trim();
      const result = $('#fillna-result');
      result.textContent = '실행 중...';
      if (!source) { result.textContent = 'Source ID를 입력하세요'; return; }
      const body = { strategy };
      if (cols) body.columns = cols.split(',').map(s => s.trim()).filter(Boolean);
      if (strategy === 'value') body.value = value;
      if (out) body.out_id = out;
      try {
        const data = await fetchJSON(`${apiBase}/datasets/${encodeURIComponent(source)}/fillna`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
        });
        result.textContent = `완료: ${data.dataset_id} (filled_total=${data.filled_total})`;
        await refreshList();
        $('#preview-id').value = data.dataset_id;
        $('#describe-id').value = data.dataset_id;
        $('#nulls-id').value = data.dataset_id;
        $('#cast-source').value = data.dataset_id;
        $('#fillna-source').value = data.dataset_id;
        $('#drop-source').value = data.dataset_id;
        $('#rename-source').value = data.dataset_id;
      } catch (e) {
        result.textContent = `실패: ${e.message}`;
      }
    });

    // drop
    $('#drop-form').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const source = $('#drop-source').value.trim();
      const cols = $('#drop-cols').value.trim();
      const out = $('#drop-out').value.trim();
      const result = $('#drop-result');
      result.textContent = '실행 중...';
      if (!source || !cols) { result.textContent = 'Source/Columns 입력'; return; }
      const body = { columns: cols.split(',').map(s => s.trim()).filter(Boolean) };
      if (out) body.out_id = out;
      try {
        const data = await fetchJSON(`${apiBase}/datasets/${encodeURIComponent(source)}/drop`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
        });
        result.textContent = `완료: ${data.dataset_id} (dropped=${(data.dropped||[]).join(',')})`;
        await refreshList();
        $('#preview-id').value = data.dataset_id;
        $('#describe-id').value = data.dataset_id;
        $('#nulls-id').value = data.dataset_id;
        $('#cast-source').value = data.dataset_id;
        $('#fillna-source').value = data.dataset_id;
        $('#drop-source').value = data.dataset_id;
        $('#rename-source').value = data.dataset_id;
      } catch (e) {
        result.textContent = `실패: ${e.message}`;
      }
    });

    // rename
    $('#rename-form').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const source = $('#rename-source').value.trim();
      const mappingStr = $('#rename-mapping').value.trim();
      const out = $('#rename-out').value.trim();
      const result = $('#rename-result');
      result.textContent = '실행 중...';
      if (!source || !mappingStr) { result.textContent = 'Source/Mapping 입력'; return; }
      const mapping = {};
      mappingStr.split(',').map(x => x.trim()).filter(Boolean).forEach(pair => {
        const [k, v] = pair.split(':');
        if (k && v) mapping[k.trim()] = v.trim();
      });
      const body = { mapping };
      if (out) body.out_id = out;
      try {
        const data = await fetchJSON(`${apiBase}/datasets/${encodeURIComponent(source)}/rename`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
        });
        result.textContent = `완료: ${data.dataset_id}`;
        await refreshList();
        $('#preview-id').value = data.dataset_id;
        $('#describe-id').value = data.dataset_id;
        $('#nulls-id').value = data.dataset_id;
        $('#cast-source').value = data.dataset_id;
        $('#fillna-source').value = data.dataset_id;
        $('#drop-source').value = data.dataset_id;
        $('#rename-source').value = data.dataset_id;
      } catch (e) {
        result.textContent = `실패: ${e.message}`;
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
    bindDistribution();
    bindCorrelation();
    bindSampleFilter();
    bindPreprocess();
    await loadInfo();
    await refreshList();
    // Autofill IDs to new sections when list refresh set preview/describe IDs
    const syncIds = () => {
      $('#dist-id').value = $('#preview-id').value;
      $('#corr-id').value = $('#preview-id').value;
      $('#nulls-id').value = $('#preview-id').value;
      $('#cast-source').value = $('#preview-id').value;
    };
    // Initial sync after brief delay
    setTimeout(syncIds, 200);
  });
})();
