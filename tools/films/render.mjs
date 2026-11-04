#!/usr/bin/env node
// Capture the book's Svelte figures at explicit clocks. No recording or second renderer.
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import os from 'node:os';
import { promises as fs, existsSync } from 'node:fs';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import assert from 'node:assert/strict';

const root = fileURLToPath(new URL('../../', import.meta.url));
const require = createRequire(path.join(root, 'print/package.json'));
const { chromium } = process.env.PLAYWRIGHT_MODULE
  ? await import(process.env.PLAYWRIGHT_MODULE) : require('playwright');
const FPS = 30, WIDTH = 1920, HEIGHT = 1080;
const films = {
  'fig-forward-backward': { seconds: 12, hold: 1.2 },
  'fig-train-live': { seconds: 16, hold: 1.5 },
  'fig-attention-triangle': { seconds: 10, hold: 1.2 },
  'fig-energy-ladder': { seconds: 10, hold: 1.2 }
};
const args = process.argv.slice(2);
const stills = args.includes('--stills');
const ids = args.filter((a) => a !== '--stills');
if (!ids.length) ids.push(...Object.keys(films));
for (const id of ids) if (!films[id]) throw Error(`Unknown film ${id}; choose ${Object.keys(films).join(', ')}`);
const output = path.resolve(process.env.FILM_OUTPUT ?? path.join(root, 'media/films'));
const review = process.env.FILM_REVIEW_DIR ?? await fs.mkdtemp(path.join(os.tmpdir(), 'picogpt-films-'));
await fs.mkdir(output, { recursive: true });
await fs.mkdir(review, { recursive: true });

async function command(cmd, argv, options = {}) {
  const child = spawn(cmd, argv, { cwd: root, stdio: 'inherit', ...options });
  const [code] = await once(child, 'close');
  if (code !== 0) throw Error(`${cmd} exited ${code}`);
}

let server, browser;
try {
  let base = process.env.BOOK_URL;
  if (!base) {
    await command('npm', ['run', 'build'], { cwd: path.join(root, 'web'), env: { ...process.env, BOOK_STRICT: '1' } });
    const build = path.join(root, 'web/build');
    const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.woff2': 'font/woff2', '.svg': 'image/svg+xml', '.json': 'application/json', '.png': 'image/png', '.webp': 'image/webp' };
    server = createServer(async (req, res) => {
      try {
        let file = path.resolve(build, '.' + decodeURIComponent(new URL(req.url, 'http://localhost').pathname));
        if (!file.startsWith(build + path.sep) && file !== build) throw Error('Outside build');
        if ((await fs.stat(file)).isDirectory()) file = path.join(file, 'index.html');
        res.setHeader('Content-Type', mime[path.extname(file)] ?? 'application/octet-stream');
        res.end(await fs.readFile(file));
      } catch { res.writeHead(404).end('Not found'); }
    });
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    base = `http://127.0.0.1:${server.address().port}/`;
  }
  browser = await chromium.launch({ headless: true });
  for (const id of ids) {
    const errors = [];
    const context = await browser.newContext({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 2, colorScheme: 'light', reducedMotion: 'no-preference' });
    const page = await context.newPage();
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
    page.on('response', (r) => { if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`); });
    await page.goto(new URL(`figures/${id}/?film&t=0`, base).href, { waitUntil: 'networkidle' });
    await page.waitForSelector('html[data-figure-ready="1"]');
    const geometry = await page.evaluate(async () => {
      document.documentElement.dataset.ink = 'paper';
      await document.fonts.ready;
      if (!document.fonts.check('16px Literata') || !document.fonts.check('16px Archivo')) throw Error('Book fonts not ready');
      const figure = document.querySelector('.fig__body');
      const height = figure.getBoundingClientRect().height;
      const available = 720 - figure.getBoundingClientRect().top - 26;
      const scale = Math.min(1, available / height);
      figure.style.transformOrigin = 'top center';
      figure.style.transform = `scale(${scale})`;
      return { width: figure.offsetWidth, height: figure.offsetHeight, viewBox: figure.querySelector('svg').getAttribute('viewBox') };
    });
    const frame = async (t) => {
      await page.evaluate((t) => window.__pico.setT(t), t);
      return page.screenshot({ type: 'png', animations: 'disabled' });
    };
    // Seek out of order: catches accumulated animation, asynchronous training and CSS easing.
    const a = await frame(0.37);
    await frame(0.81);
    const b = await frame(0.37);
    assert(a.equals(b), `${id}: seeking is not deterministic`);
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      const png = await frame(t);
      const current = await page.evaluate(() => {
        const figure = document.querySelector('.fig__body');
        return { width: figure.offsetWidth, height: figure.offsetHeight, viewBox: figure.querySelector('svg').getAttribute('viewBox') };
      });
      assert.deepEqual(current, geometry, `${id}: figure layout changed at t=${t}`);
      await fs.writeFile(path.join(review, `${id}-${String(t * 100).padStart(3, '0')}.png`), png);
    }
    for (const selector of ['.scrub', '.readout', '.fx-live__controls', '.fx-ladder__reset']) {
      assert(!(await page.locator(selector).first().isVisible()), `${id}: interaction chrome is visible (${selector})`);
    }
    if (id === 'fig-train-live') assert.equal(await page.locator('.fx-live').getAttribute('data-step'), '200');
    if (errors.length) throw Error(`${id}: ${errors.join('\n')}`);
    if (!stills) {
      const { seconds, hold } = films[id];
      const ffmpeg = process.env.FFMPEG ?? (existsSync(path.join(os.homedir(), '.local/bin/ffmpeg')) ? path.join(os.homedir(), '.local/bin/ffmpeg') : 'ffmpeg');
      const child = spawn(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'image2pipe', '-framerate', String(FPS), '-i', 'pipe:0',
        '-filter_complex', `[0:v]scale=${WIDTH}:${HEIGHT}:flags=lanczos,split=2[mp4][webm]`,
        '-map', '[mp4]', '-an', '-c:v', 'libx264', '-threads', '4', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-map_metadata', '-1', path.join(output, `${id}.mp4`),
        '-map', '[webm]', '-an', '-c:v', 'libvpx-vp9', '-threads', '4', '-b:v', '0', '-crf', '32', '-row-mt', '1', '-cpu-used', '3', '-pix_fmt', 'yuv420p', '-map_metadata', '-1', path.join(output, `${id}.webm`)], { stdio: ['pipe', 'inherit', 'inherit'] });
      const completion = new Promise((resolve, reject) => { child.once('error', reject); child.once('close', (code) => code === 0 ? resolve() : reject(Error(`ffmpeg exited ${code}`))); });
      // Handle errors immediately while captures are being fed to the encoder.
      completion.catch(() => {});
      let lastT = -1, png;
      try {
        for (let i = 0; i < seconds * FPS; i++) {
          const t = Math.max(0, Math.min(1, (i / FPS - hold) / (seconds - 2 * hold - 1 / FPS)));
          if (t !== lastT) { png = await frame(t); lastT = t; }
          await new Promise((resolve, reject) => child.stdin.write(png, (err) => err ? reject(err) : resolve()));
          if (i % 90 === 0) console.log(`${id}: ${i}/${seconds * FPS} frames`);
        }
        child.stdin.end();
        await completion;
      } catch (e) { child.kill('SIGTERM'); throw e; }
      // Same final frame, downsampled by the same filter as the films.
      const poster = spawn(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'image2pipe', '-i', 'pipe:0', '-vf', `scale=${WIDTH}:${HEIGHT}:flags=lanczos`, '-frames:v', '1', path.join(output, `${id}.png`)], { stdio: ['pipe', 'inherit', 'inherit'] });
      const posterDone = once(poster, 'close');
      poster.stdin.end(await frame(1));
      assert.equal((await posterDone)[0], 0, 'Poster encoding failed');
      for (const ext of ['mp4', 'webm', 'png']) console.log(`${path.join(output, `${id}.${ext}`)} (${(await fs.stat(path.join(output, `${id}.${ext}`))).size} bytes)`);
      if (errors.length) throw Error(`${id}: ${errors.join('\n')}`);
    }
    await context.close();
  }
  console.log(`Review frames (five per film, 2560×1440): ${review}`);
} finally {
  await browser?.close();
  if (server) await new Promise((resolve) => server.close(resolve));
}
