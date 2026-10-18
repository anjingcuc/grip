import { chromium } from "playwright-core";
const browser = await chromium.connectOverCDP("http://127.0.0.1:9222");
const page = browser.contexts()[0].pages()[0];
await page.screenshot({ path: "tools/e2e/anim_a.png" });
await page.waitForTimeout(1300);
await page.screenshot({ path: "tools/e2e/anim_b.png" });
console.log("ok");
process.exit(0);
