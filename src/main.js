import '@fontsource/ibm-plex-mono/latin-400.css';
import '@fontsource/ibm-plex-mono/latin-500.css';
import './styles.css';
import './entrance.css';

const body = document.body;
const site = document.querySelector('#site');
const sculpture = document.querySelector('#sculpture');
const pepperField = document.querySelector('#pepper-field');
const start = document.querySelector('#start-lab');
const ripple = document.querySelector('#ripple-button');
const status = document.querySelector('#entry-status');
const labRoot = document.querySelector('#lab-root');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const landingTitle = document.title;
const pagesPreview = import.meta.env.MODE === 'pages';
if (pagesPreview) {
  start.href = 'https://github.com/Aditya-garg02/facial-expression-lab#quick-start';
  start.textContent = 'Run the lab locally';
}
const labPath = new URL(start.href).pathname;
const sculptureAnchor = document.createComment('Sculpture mount');
const entryLayer = document.createElement('div');

sculpture.before(sculptureAnchor);
entryLayer.className = 'entry-motion';
entryLayer.setAttribute('aria-hidden', 'true');

let scene;
let entry;

function startSculpture() {
  if (scene || entry) return;
  import('./scene.js').then(({ createSculpture }) => {
    if (scene || entry) return;
    scene = createSculpture(sculpture, pepperField);
    ripple.hidden = typeof scene.pulse !== 'function';
  }).catch((error) => {
    console.warn('The point cloud could not start.', error);
  });
}

function displayBenchmark() {
  const reports = import.meta.glob('../artifacts/metrics.json', { eager: true, import: 'default' });
  const metrics = reports['../artifacts/metrics.json'];
  if (!metrics) return;

  const values = document.querySelector('#report-values');
  for (const label of metrics.classes) {
    const recall = (100 * metrics.per_class[label].recall).toFixed(1);
    const row = document.createElement('li');
    const name = document.createElement('span');
    name.textContent = label;
    const track = document.createElement('span');
    track.className = 'report__track';
    track.setAttribute('aria-hidden', 'true');
    const fill = document.createElement('span');
    fill.className = 'report__fill';
    fill.style.setProperty('--recall', `${recall}%`);
    const value = document.createElement('span');
    value.className = 'report__value';
    value.textContent = `${recall}%`;
    track.append(fill);
    row.append(name, track, value);
    values.append(row);
  }
  document.querySelector('#report-preview').hidden = false;
}

function waitForWorkbench(frame, signal) {
  return new Promise((resolve) => {
    const deadline = performance.now() + 35_000;
    let timer;
    function finish(ready) {
      clearTimeout(timer);
      signal.removeEventListener('abort', cancel);
      resolve(ready);
    }
    function cancel() {
      finish(false);
    }
    function check() {
      const content = frame.contentDocument;
      if (content?.getElementById('lab-workbench') || content?.querySelector('[data-testid="stException"]')) {
        finish(true);
      } else if (performance.now() >= deadline) {
        finish(false);
      } else {
        timer = setTimeout(check, 100);
      }
    }
    signal.addEventListener('abort', cancel, { once: true });
    check();
  });
}

async function enterLab() {
  const controller = new AbortController();
  const cinematic = !reducedMotion.matches && Boolean(scene?.animated);
  const duration = cinematic ? scene.enterDuration ?? 1040 : 150;
  entry = controller;

  const origin = cinematic ? sculpture.getBoundingClientRect() : null;
  entryLayer.classList.toggle('entry-motion--cinematic', cinematic);
  entryLayer.style.setProperty('--entry-duration', `${duration}ms`);
  body.append(entryLayer);
  body.classList.add('is-entering');
  site.inert = true;
  if (cinematic) {
    entryLayer.append(sculpture);
    scene.enter(origin);
  }

  const frame = document.createElement('iframe');
  frame.className = 'lab-frame';
  frame.title = 'ExpressionLab workbench';
  frame.allow = 'camera';
  frame.src = labPath;
  frame.setAttribute('aria-hidden', 'true');
  frame.tabIndex = -1;
  labRoot.append(frame);

  const loading = setTimeout(() => {
    status.textContent = 'Opening the lab…';
  }, 1300);
  controller.signal.addEventListener('abort', () => {
    clearTimeout(loading);
  }, { once: true });

  const [ready] = await Promise.all([
    waitForWorkbench(frame, controller.signal),
    new Promise((resolve) => setTimeout(resolve, duration)),
  ]);
  if (controller.signal.aborted) return;
  if (!ready) {
    location.replace(labPath);
    return;
  }

  clearTimeout(loading);
  status.textContent = '';
  frame.removeAttribute('aria-hidden');
  frame.removeAttribute('tabindex');
  body.classList.replace('is-entering', 'is-in-lab');
  document.title = 'ExpressionLab · Workbench';
  frame.focus({ preventScroll: true });
  releaseSculpture();
}

function releaseSculpture() {
  scene?.destroy();
  scene = undefined;
  if (sculpture.parentElement === entryLayer) sculptureAnchor.after(sculpture);
}

function returnToLanding() {
  entry?.abort();
  entry = undefined;
  releaseSculpture();
  entryLayer.remove();
  labRoot.replaceChildren();
  body.classList.remove('is-entering', 'is-in-lab');
  site.inert = false;
  status.textContent = '';
  document.title = landingTitle;
  startSculpture();
  start.focus({ preventScroll: true });
}

start.addEventListener('click', (event) => {
  if (pagesPreview) return;
  if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  if (entry) return;
  history.pushState({ workbench: true }, '', labPath);
  enterLab();
});

window.addEventListener('popstate', (event) => {
  if (event.state?.workbench && location.pathname === labPath) {
    if (!entry) enterLab();
  } else if (entry) {
    returnToLanding();
  }
});

ripple.addEventListener('click', () => scene?.pulse?.());
displayBenchmark();
startSculpture();
