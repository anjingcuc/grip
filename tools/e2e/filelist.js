// 文件列表行高验证：截侧栏 + 量测各行实际高度
import { chromium } from 'playwright-core';
const b = await chromium.connectOverCDP('http://127.0.0.1:9222');
const p = b.contexts()[0].pages()[0];
await p.waitForTimeout(1200);
const info = await p.$$eval('#file-list .list-item', (els) => {
  const r = els.map((e) => Math.round(e.getBoundingClientRect().height));
  return { count: els.length, heights: r.slice(0, 6), scroll: els[0]?.parentElement?.scrollHeight };
});
console.log(JSON.stringify(info));
await p.screenshot({ path: 'filelist_fixed.png' });
process.exit(0);
