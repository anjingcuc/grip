// 描边滑杆 GUI 验证：开图 → 选线 → 描边拉满 → 截结果态
import { chromium } from 'playwright-core';
const b = await chromium.connectOverCDP('http://127.0.0.1:9222');
const page = b.contexts()[0].pages()[0];
await page.waitForTimeout(1000);
const first = await page.$('.file-item');
if (first) {
  await first.click();
  await page.waitForSelector('#scan-overlay[hidden]', { timeout: 60000 }).catch(() => {});
  await page.waitForTimeout(800);
}
const items = await page.$$('#cluster-list .list-item:not(.list-head)');
if (items.length > 1) { await items[1].click(); await page.waitForTimeout(900); }
await page.$eval('#p-outline', (el) => { el.value = '1'; el.dispatchEvent(new Event('change', { bubbles: true })); });
await page.waitForTimeout(2500);
const v = await page.$eval('#p-outline', (el) => el.value);
const label = await page.$eval('#p-outline-v', (el) => el.textContent);
console.log('outline =', v, 'label =', label);
await page.screenshot({ path: 'outline_gui.png' });
process.exit(0);
