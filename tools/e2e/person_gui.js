// 人物保色 GUI 验证：读模型状态行 + 滑杆存在性 + 截图
import { chromium } from 'playwright-core';
const b = await chromium.connectOverCDP('http://127.0.0.1:9222');
const p = b.contexts()[0].pages()[0];
await p.setViewportSize({ width: 1280, height: 900 }).catch(() => {});
await p.waitForTimeout(1500);
const status = await p.$eval('#model-status', (e) => e.textContent);
console.log('model-status:', status);
const hasSlider = (await p.$('#p-person')) != null;
console.log('person slider present:', hasSlider);
await p.screenshot({ path: 'person_gui.png' });
process.exit(0);
