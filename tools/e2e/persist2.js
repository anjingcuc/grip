// 持久化验证·下半场：重开同一张图片，滑杆应恢复 0.80 且画面带净化
// 用法: node persist2.js <图片路径>
import { chromium } from 'playwright-core';
const target = process.argv[2];
const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
const page = browser.contexts()[0].pages()[0];
await page.waitForTimeout(1200);
const items = await page.$$('.file-item');
let item = null;
for (const it of items) {
  if ((await it.getAttribute('title')) === target) { item = it; break; }
}
if (!item) { console.error('file item not found: ' + target + '（共 ' + items.length + ' 项）'); process.exit(1); }
await item.click();
// 等分析/会话恢复完成（扫描遮罩隐藏）
await page.waitForSelector('#scan-overlay[hidden]', { timeout: 60000 }).catch(() => {});
await page.waitForTimeout(1500);
const v = await page.$eval('#p-bgfill', (el) => el.value);
console.log('restored bgfill =', v);
await page.screenshot({ path: 'persist_restored.png' });
process.exit(v === '0.8' ? 0 : 2);
