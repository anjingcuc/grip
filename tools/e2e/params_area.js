// 参数区整理验证：开关行为 + 高级参数折叠 + 布局截图
import { chromium } from 'playwright-core';
const b = await chromium.connectOverCDP('http://127.0.0.1:9222');
const page = b.contexts()[0].pages()[0];
await page.setViewportSize({ width: 1280, height: 900 }).catch(() => {});
await page.waitForTimeout(1200);
const first = await page.$('.file-item');
if (first) {
  await first.click();
  await page.waitForSelector('#scan-overlay[hidden]', { timeout: 60000 }).catch(() => {});
  await page.waitForTimeout(800);
}
const items = await page.$$('#cluster-list .list-item:not(.list-head)');
if (items.length > 1) { await items[1].click(); await page.waitForTimeout(900); }

// 折叠区默认收起
const advClosed = await page.$eval('#adv-params', (el) => !el.open);
console.log('adv closed by default:', advClosed);

// 开关默认开（person_keep=1），关掉应触发预览重渲
const checked0 = await page.$eval('#p-person', (el) => el.checked);
async function previewHash() {
  return page.evaluate(() => {
    const s = document.querySelector('#view-img')?.src || '';
    let h = 0;
    for (let i = 0; i < s.length; i += 997) h = (h * 31 + s.charCodeAt(i)) >>> 0;
    return { len: s.length, h };
  });
}
const hOn = await previewHash();
await page.$eval('.switch', (el) => el.click());
await page.waitForTimeout(2200);
const checked1 = await page.$eval('#p-person', (el) => el.checked);
const hOff = await previewHash();
console.log('switch', checked0, '->', checked1, '| preview changed:', hOn.len !== hOff.len || hOn.h !== hOff.h);

// 展开高级参数再截图
await page.$eval('#adv-params > summary', (el) => el.click());
await page.waitForTimeout(400);
await page.screenshot({ path: 'params_area.png' });
process.exit(0);
