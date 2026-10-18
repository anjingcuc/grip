import { chromium } from "playwright-core";
const browser = await chromium.connectOverCDP("http://127.0.0.1:9222");
const page = browser.contexts()[0].pages()[0];
await page.waitForTimeout(1200);
await page.screenshot({ path: "tools/e2e/empty_state.png" });
console.log("ok");
process.exit(0);
