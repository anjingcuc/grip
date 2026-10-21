// 开关开态验证：computed 样式应为 accent 橙 + 预览重渲
import { chromium } from 'playwright-core';
const b = await chromium.connectOverCDP('http://127.0.0.1:9222');
const page = b.contexts()[0].pages()[0];
await page.setViewportSize({ width: 1280, height: 900 }).catch(() => {});
await page.waitForTimeout(1200);
const first = await page.$('.file-item');
if (first) {
  await first.click();
  await page.waitForSelector('#scan-overlay[hidden]', { timeout: 60000 }).catch(() => {});
  await page.waitForTimeout(600);
}
// 打开开关（若未开）
const wasChecked = await page.$eval('#p-person', (el) => el.checked);
if (!wasChecked) await page.$eval('.switch', (el) => el.click());
await page.waitForTimeout(2200);
const nowChecked = await page.$eval('#p-person', (el) => el.checked);
const trackBg = await page.$eval('.switch .track', (el) => getComputedStyle(el).backgroundColor);
const knobX = await page.$eval('.switch .track', (el) => getComputedStyle(el, '::after').transform);
console.log('checked:', nowChecked, '| track bg:', trackBg, '| knob transform:', knobX);
await page.screenshot({ path: 'params_switch_on.png' });
process.exit(0);
