// CDP 截图脚本：连到 Tauri WebView2 截屏
// 用法: node shot.js <输出前缀> [等待毫秒]
import { chromium } from 'playwright-core';

const prefix = process.argv[2] || 'shot';
const waitMs = parseInt(process.argv[3] || '1200', 10);
const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
const ctx = browser.contexts()[0];
const pages = ctx.pages();
if (!pages.length) { console.error('no pages'); process.exit(1); }
const page = pages[0];
await page.waitForTimeout(waitMs);
await page.screenshot({ path: `${prefix}.png` });
// 再点一下第一个颜色组（跳过列头与"无"行），截结果态
try {
  const items = await page.$$('#cluster-list .list-item:not(.list-head)');
  if (items.length > 1) {
    await items[1].click();
    await page.waitForTimeout(900);
    await page.screenshot({ path: `${prefix}_selected.png` });
  }
} catch (e) { console.error('click cluster failed:', e.message); }
console.log('done');
process.exit(0);
