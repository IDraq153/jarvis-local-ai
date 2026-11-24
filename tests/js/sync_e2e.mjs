// Prueba de punta a punta de la sincronización de store.js (web real) con el servidor real de JARVIS.
// Lo lanza tests/test_unigym_web_sync.py; variables: UNIGYM_WEB, PORT, MODE=normal|nofile. Datos inventados.
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';

const PORT = process.env.PORT;
const MODE = process.env.MODE || 'normal';
const base = `http://127.0.0.1:${PORT}`;
const realFetch = globalThis.fetch;
globalThis.fetch = (u, o) => realFetch(u.startsWith('/') ? base + u : u, o);
const mem = new Map();
globalThis.localStorage = { getItem: (k) => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, String(v)) };
globalThis.location = { protocol: 'http:' };
globalThis.window = new EventTarget();
globalThis.document = { activeElement: null, hidden: false, getElementById: () => null };
const events = [];
window.addEventListener('unigym:sync', (e) => events.push(e.detail.kind));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const store = await import(pathToFileURL(`${process.env.UNIGYM_WEB}/js/core/store.js`).href);
const api = async (method, body) => {
  const r = await realFetch(`${base}/api/state`, {
    method, headers: { 'Content-Type': 'application/json', 'X-UniGym': '1' }, body: body ? JSON.stringify(body) : undefined });
  return { status: r.status, json: await r.json() };
};

if (MODE === 'nofile') {
  assert.equal(await store.connectRemote({ pollMs: 0 }), false, 'sin archivo no debe conectar');
  const { firstRun } = store.initStore(() => ({ tasks: [{ id: 'semilla', title: 'Semilla', due: '', done: false }] }));
  assert.equal(firstRun, true);
  store.update((s) => s.tasks.push({ id: 'local', title: 'Solo local', due: '', done: false }));
  await sleep(300);
  assert.equal(JSON.parse(mem.get(store.STORAGE_KEY)).tasks.length, 2, 'sigue guardando en localStorage');
  console.log('OK nofile');
  process.exit(0);
}

// 1. conecta y carga los datos compartidos
assert.equal(await store.connectRemote({ pollMs: 50 }), true);
const { firstRun } = store.initStore(() => assert.fail('no debe usar la semilla'));
assert.equal(firstRun, false);
assert.equal(store.getState().tasks[0].title, 'Tarea uno');
assert.ok(mem.get(store.STORAGE_KEY), 'localStorage queda como caché');

// 2. un cambio en la web llega al archivo
store.update((s) => s.tasks.push({ id: 'tw', title: 'Desde la web', due: '', done: false }));
await sleep(400);
let cur = (await api('GET')).json;
assert.equal(cur.revision, 2);
assert.equal(cur.by, 'web');
assert.ok(cur.data.tasks.some((t) => t.id === 'tw'));

// 3. un cambio externo (JARVIS) aparece en la web y la vista se entera
let notified = 0;
store.subscribe(() => { notified += 1; });
cur.data.settings.externo = 1;
assert.equal((await api('PUT', { baseRevision: cur.revision, data: cur.data })).status, 200);
await sleep(300);
assert.equal(store.getState().settings.externo, 1);
assert.ok(notified >= 1);
assert.ok(events.includes('remote'));

// 4. si estás escribiendo en un campo, no te pisa lo que tecleas: espera
document.activeElement = { tagName: 'INPUT' };
cur = (await api('GET')).json;
cur.data.settings.externo = 2;
await api('PUT', { baseRevision: cur.revision, data: cur.data });
await sleep(300);
assert.equal(store.getState().settings.externo, 1, 'no debe aplicar mientras editas');
document.activeElement = null;
await sleep(300);
assert.equal(store.getState().settings.externo, 2, 'se aplica al terminar de editar');

// 5. conflicto: JARVIS escribe y la web guarda sin haberlo visto -> gana el archivo, aviso 'conflict'
document.hidden = true; // pausa el sondeo para provocar la carrera
cur = (await api('GET')).json;
cur.data.settings.externo = 3;
await api('PUT', { baseRevision: cur.revision, data: cur.data });
store.update((s) => { s.settings.soloWeb = true; });
await sleep(400);
assert.equal(store.getState().settings.externo, 3, 'adopta lo del archivo');
assert.equal(store.getState().settings.soloWeb, undefined, 'el cambio perdido se avisa, no se mezcla');
assert.ok(events.includes('conflict'));
document.hidden = false;

// 6. tras el conflicto la web puede volver a guardar
store.update((s) => { s.settings.despues = 'ok'; });
await sleep(400);
assert.equal((await api('GET')).json.data.settings.despues, 'ok');

store.disconnectRemote();
console.log('OK normal');
process.exit(0);
