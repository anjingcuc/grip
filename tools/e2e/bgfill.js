// CDP 验证：墙面净化滑杆驱动预览实时更新
// 用法: node bgfill.js <输出前缀>
import { chromium } from 'playwright-core';

const prefix = process.argv[2] || 'bgfill';
const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
const page = browser.contexts()[0].pages()[0];
if (!page) { console.error('no pages'); process.exit(1); }
await page.waitForTimeout(1000);

// 0) 初始态（滑杆应为 0）
await page.screenshot({ path: `${prefix}_0.png` });

// 1) 点第一个颜色组进入结果态
const items = await page.$$('#cluster-list .list-item:not(.list-head)');
if (items.length > 1) { await items[1].click(); await page.waitForTimeout(900); }

// 2) 拖净化滑杆到 0.8 并触发 change（与 UI 绑定路径一致）
await page.$eval('#p-bgfill', (el) => {
  el.value = '0.8';
  el.dispatchEvent(new Event('change', { bubbles: true }));
});
await page.waitForTimeout(2500);
await page.screenshot({ path: `${prefix}_80.png` });
const v = await page.$eval('#p-bgfill-v', (el) => el.textContent);
console.log('bgfill label =', v);

// 3) 回到 0
await page.$eval('#p-bgfill', (el) => {
  el.value = '0';
  el.dispatchEvent(new Event('change', { bubbles: true }));
});
await page.waitForTimeout(2000);
await page.screenshot({ path: `${prefix}_back0.png` });
console.log('done');
process.exit(0);
