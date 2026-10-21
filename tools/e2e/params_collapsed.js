// 参数区收起态截图（先合上高级参数，清记忆）
import { chromium } from 'playwright-core';
const b = await chromium.connectOverCDP('http://127.0.0.1:9222');
const page = b.contexts()[0].pages()[0];
await page.setViewportSize({ width: 1280, height: 900 }).catch(() => {});
await page.waitForTimeout(1200);
await page.evaluate(() => localStorage.removeItem('adv-params-open'));
const open = await page.$eval('#adv-params', (el) => el.open);
if (open) await page.$eval('#adv-params > summary', (el) => el.click());
await page.waitForTimeout(400);
const first = await page.$('.file-item');
if (first) {
  await first.click();
  await page.waitForSelector('#scan-overlay[hidden]', { timeout: 60000 }).catch(() => {});
  await page.waitForTimeout(600);
}
await page.screenshot({ path: 'params_collapsed.png' });
process.exit(0);
