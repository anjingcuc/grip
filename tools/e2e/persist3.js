// 闭环验证：重开图片 → 点回选中的颜色组 → 结果态截图（与 bgfill_80 对照）
import { chromium } from 'playwright-core';
const target = process.argv[2];
const b = await chromium.connectOverCDP('http://127.0.0.1:9222');
const page = b.contexts()[0].pages()[0];
await page.waitForTimeout(1200);
const items = await page.$$('.file-item');
for (const it of items) {
  if ((await it.getAttribute('title')) === target) { await it.click(); break; }
}
await page.waitForSelector('#scan-overlay[hidden]', { timeout: 60000 }).catch(() => {});
await page.waitForTimeout(1200);
// 恢复会话里已有选中组：点 active 的颜色组项重新渲染结果预览
const active = await page.$('#cluster-list .list-item.active:not(.list-head)');
const pick = active || (await page.$$('#cluster-list .list-item:not(.list-head)'))[1];
if (pick) await pick.click();
await page.waitForTimeout(2500);
const v = await page.$eval('#p-bgfill', (el) => el.value);
console.log('bgfill =', v);
await page.screenshot({ path: 'persist_result.png' });
process.exit(0);
