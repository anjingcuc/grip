// 描边 GUI 复核：大视口 + 预览字节哈希对比（outline 0 vs 1 必须不同）
import { chromium } from 'playwright-core';
const b = await chromium.connectOverCDP('http://127.0.0.1:9222');
const page = b.contexts()[0].pages()[0];
await page.setViewportSize({ width: 1280, height: 900 }).catch((e) => console.error('viewport:', e.message));
await page.waitForTimeout(800);
const first = await page.$('.file-item');
if (first) {
  await first.click();
  await page.waitForSelector('#scan-overlay[hidden]', { timeout: 60000 }).catch(() => {});
  await page.waitForTimeout(800);
}
const items = await page.$$('#cluster-list .list-item:not(.list-head)');
if (items.length > 1) { await items[1].click(); await page.waitForTimeout(900); }

async function setOutline(v) {
  await page.$eval('#p-outline', (el, val) => { el.value = val; el.dispatchEvent(new Event('change', { bubbles: true })); }, String(v));
  await page.waitForTimeout(2200);
}
async function previewHash() {
  return page.evaluate(() => {
    const img = document.querySelector('#view-img');
    const s = img ? img.src : '';
    let h = 0;
    for (let i = 0; i < s.length; i += 997) h = (h * 31 + s.charCodeAt(i)) >>> 0;
    return { len: s.length, h };
  });
}
await setOutline(0);
const h0 = await previewHash();
await setOutline(1);
const h1 = await previewHash();
console.log('outline0:', JSON.stringify(h0));
console.log('outline1:', JSON.stringify(h1));
console.log(h0.len !== h1.len || h0.h !== h1.h ? 'PREVIEW CHANGED ✓' : 'PREVIEW UNCHANGED ✗');
await page.screenshot({ path: 'outline_gui2.png' });
process.exit(0);
