// 持久化验证·上半场：记录当前图片，设净化 0.8（触发会话落盘）
import { chromium } from 'playwright-core';
const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
const page = browser.contexts()[0].pages()[0];
await page.waitForTimeout(800);
let title = await page.$eval('.file-item.active', (el) => el.title).catch(() => null);
if (!title) {
  const first = await page.$('.file-item');
  if (!first) { console.error('no file items'); process.exit(1); }
  title = await first.getAttribute('title');
  await first.click();
  await page.waitForTimeout(12000);
}
await page.$eval('#p-bgfill', (el) => { el.value = '0.8'; el.dispatchEvent(new Event('change', { bubbles: true })); });
await page.waitForTimeout(1500);
console.log('IMAGE=' + title);
process.exit(0);
