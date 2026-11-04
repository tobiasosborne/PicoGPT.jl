// Screenshot loop for design review: node scripts/shots.mjs <path> [name] [--full] [--mobile] [--dark] [--reduced] [--w=1440] [--h=900] [--y=0] [--print]
// Requires Playwright's Chromium (global `playwright` package or PLAYWRIGHT_MODULE path).
import path from 'node:path';
import fs from 'node:fs';
const PW = process.env.PLAYWRIGHT_MODULE ?? '/home/tobias/.nvm/versions/node/v24.11.1/lib/node_modules/playwright/index.mjs';
const { chromium } = await import(PW);
const args = process.argv.slice(2);
const url = new URL(args[0] ?? '/', process.env.BOOK_URL ?? 'http://localhost:5173').href;
const name = args[1] && !args[1].startsWith('--') ? args[1] : 'shot';
const flag = (f) => args.includes(`--${f}`);
const opt = (k, d) => +(args.find((a) => a.startsWith(`--${k}=`))?.split('=')[1] ?? d);
const mobile = flag('mobile');
const W = mobile ? 390 : opt('w', 1440), H = mobile ? 844 : opt('h', 900);
const out = process.env.SHOTS ?? '/tmp/picodesign/shots';
fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch();
const context = await browser.newContext({
	viewport: { width: W, height: H }, deviceScaleFactor: opt('dpr', mobile ? 2 : 1),
	colorScheme: flag('dark') ? 'dark' : 'light', reducedMotion: flag('reduced') ? 'reduce' : 'no-preference'
});
const page = await context.newPage();
const errors = [];
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') errors.push(`${m.type()}: ${m.text()}`); });
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
if (flag('print')) await page.emulateMedia({ media: 'print' });
await page.goto(url, { waitUntil: 'networkidle', timeout: 60000 });
await page.evaluate(() => document.fonts.ready);
await page.waitForTimeout(opt('wait', 600));
const y = opt('y', 0);
if (y) { await page.evaluate((y) => window.scrollTo(0, y), y); await page.waitForTimeout(300); }
const sel = args.find((a) => a.startsWith('--el='))?.slice(5);
const file = path.join(out, `${name}.png`);
if (sel) await page.locator(sel).first().screenshot({ path: file });
else await page.screenshot({ path: file, fullPage: flag('full') });
console.log(file, errors.length ? `\n${errors.join('\n')}` : '');
await browser.close();
