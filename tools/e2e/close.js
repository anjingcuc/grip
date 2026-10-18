import { chromium } from "playwright-core";
const browser = await chromium.connectOverCDP("http://127.0.0.1:9222");
const page = browser.contexts()[0].pages()[0];
await page.evaluate(() => window.__TAURI__.window.getCurrentWindow().close());
await page.waitForTimeout(1500);
console.log("closed via window.close()");
process.exit(0);
