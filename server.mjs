import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { createServer as createSocket } from 'node:net';
import { dirname, join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { createServer, normalizePath, preview } from 'vite';

const root = import.meta.dirname;
const host = '127.0.0.1';
const port = Number(process.env.EXPRESSIONLAB_PORT ?? 4173);
const built = process.argv.includes('--built');

function findPython() {
  if (process.env.EXPRESSIONLAB_PYTHON) return process.env.EXPRESSIONLAB_PYTHON;

  const executable = process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python';
  for (const directory of [root, dirname(root)]) {
    const localPython = join(directory, '.venv', executable);
    if (existsSync(localPython)) return localPython;
  }
  return process.platform === 'win32' ? 'python' : 'python3';
}

function getFreePort() {
  return new Promise((resolve, reject) => {
    const socket = createSocket().once('error', reject);
    socket.listen(0, host, () => {
      const { port } = socket.address();
      socket.close((error) => error ? reject(error) : resolve(port));
    });
  });
}

async function waitForLab(origin, streamlit) {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline && streamlit.exitCode === null && streamlit.signalCode === null) {
    try {
      const response = await fetch(`${origin}/lab/_stcore/health`, {
        signal: AbortSignal.timeout(650),
      });
      if (response.ok) return;
    } catch {
      // Streamlit hasn't started listening yet.
    }
    await delay(200);
  }
  throw new Error('Streamlit did not start. Check its output above and install requirements.txt in your Python environment.');
}

async function start() {
  if (!Number.isInteger(port) || port < 1 || port > 65_535) {
    throw new Error('EXPRESSIONLAB_PORT must be a valid TCP port.');
  }
  if (built && !existsSync(join(root, 'dist', 'index.html'))) {
    throw new Error('Build the landing page with npm run build before running npm start.');
  }

  const labPort = await getFreePort();
  const labOrigin = `http://${host}:${labPort}`;
  const streamlit = spawn(findPython(), [
    '-m', 'streamlit', 'run', 'app.py',
    `--server.address=${host}`,
    `--server.port=${labPort}`,
    '--server.baseUrlPath=lab',
    '--server.headless=true',
    '--browser.gatherUsageStats=false',
  ], { cwd: root, stdio: 'inherit' });
  process.once('exit', () => streamlit.kill());
  for (const signal of ['SIGINT', 'SIGTERM']) {
    process.once(signal, () => process.exit());
  }

  try {
    await Promise.race([
      waitForLab(labOrigin, streamlit),
      new Promise((_, reject) => streamlit.once('error', reject)),
    ]);

    const network = {
      host,
      port,
      strictPort: true,
      proxy: {
        '^/lab(?:/|\\?|$)': { target: labOrigin, ws: true },
      },
    };
    const options = {
      root,
      configFile: false,
      appType: 'mpa',
      server: {
        ...network,
        fs: {
          deny: [
            '.env', '.env.*', '*.{crt,pem}', '**/.git/**',
            '**/.venv/**', '**/.streamlit/**', `${normalizePath(root)}/data/**`, '**/*.{py,pyc,pt}',
          ],
        },
      },
      preview: network,
    };
    if (built) {
      await preview(options);
    } else {
      const vite = await createServer(options);
      await vite.listen();
    }
  } catch (error) {
    streamlit.kill();
    throw error;
  }
  streamlit.once('exit', () => {
    console.error('Streamlit stopped. Restart ExpressionLab to reconnect the lab.');
    process.exit(1);
  });

  console.log(`\n  ExpressionLab   http://${host}:${port}/\n`);
}

start().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
