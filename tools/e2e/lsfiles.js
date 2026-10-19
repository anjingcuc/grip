import { chromium } from 'playwright-core';
const b = await chromium.connectOverCDP('http://127.0.0.1:9222');
const p = b.contexts()[0].pages()[0];
const t = await p.$$eval('.file-item', (els) => els.map((e) => e.title).slice(0, 5));
console.log(t.join('\n'));
const f = await p.$eval('#folder-name', (e) => e.textContent).catch(() => '?');
console.log('folder:', f);
process.exit(0);
