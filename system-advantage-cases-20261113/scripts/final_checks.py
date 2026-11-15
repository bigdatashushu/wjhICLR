"""Verify the final prose-only correction against fully browser-checked copies."""
import asyncio
import hashlib
import json
import pathlib
import shutil
import zipfile
from playwright.async_api import async_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
old = "直答选择 Left / B，把不同拍摄方向下的视觉变化作了左向判断。"
new = "直答选择 Left / B，与题目参考系下更大的后向分量不符。"
manifest = json.loads((ROOT / "build/delivery-manifest.json").read_text())
delta = []
for m in manifest:
    p = ROOT / m["file"]
    tested = ROOT / "checks/isolated" / m["file"]
    before = tested.read_text()
    after = p.read_text()
    assert before.replace(old, new) == after, m["file"]
    delta.append({"file": m["file"], "only_changed_text": old in before,
                  "runtime_code_and_images_identical": True,
                  "final_sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    shutil.copyfile(p, tested)

async def browser_check():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True)
        for name in ("index.html", "MMSI.html"):
            ctx = await browser.new_context(offline=True, viewport={"width": 1440, "height": 1050})
            page = await ctx.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            await page.goto((ROOT / "checks/isolated" / name).as_uri())
            await page.evaluate("window.__casebook.select('MMSI','MMSI_247')")
            text = await page.locator(".evidence-card").inner_text()
            assert new in text, name
            assert not errors
            await ctx.close()
        await browser.close()
asyncio.run(browser_check())
with zipfile.ZipFile(ROOT / "空间智能系统优势案例-30题-离线分享.zip") as z:
    assert z.testzip() is None
    assert set(z.namelist()) == {m["file"] for m in manifest} | {"README.txt"}
    assert all(hashlib.sha256(z.read(m["file"])).hexdigest() == m["sha256"] for m in manifest)
(ROOT / "checks/final-version-validation.json").write_text(json.dumps({
    "full_browser_validation": "offline-browser-validation.json",
    "final_delta": delta,
    "changed_pages_reopened_offline": ["index.html", "MMSI.html"],
    "zip_crc": "passed",
    "zip_html_hashes_match_manifest": True,
    "usage_note": "准确说明归档图片原字节与 VSIbench 冻结帧 RGB 像素核验的区别。"
}, ensure_ascii=False, indent=2))
print("Final version verified: seven standalone HTML files, unchanged code/images, corrected prose, ZIP CRC and hashes passed.")
