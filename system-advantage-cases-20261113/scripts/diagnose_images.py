import asyncio
import pathlib
from playwright.async_api import async_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True)
        c = await b.new_context(offline=True)
        page = await c.new_page()
        page.on("console", lambda x: print("CONSOLE", x.type, x.text[:500]))
        for name in ("BLINK.html", "index.html"):
            await page.goto((ROOT / "checks/isolated" / name).as_uri())
            result = await page.evaluate("""async()=>{const h='5c07e562f87a7f6e436b9a4184e5a7da0a247655c31cfbb1d05eee022f379d9b';const a=window.__casebook.data.assets[h];const url=window.__casebook.assetURL(h);const im=new Image();im.src=url;try{await im.decode();return {width:im.naturalWidth,height:im.naturalHeight,head:url.slice(0,80),base64_length:a.base64.length}}catch(e){return {error:e.message,head:url.slice(0,80),base64_length:a.base64.length}}}""")
            print(name, result)
            if name == "index.html":
                result = await page.evaluate("""async()=>{const failures=[];let n=0;for(const [h,a] of Object.entries(window.__casebook.data.assets)){const im=new Image();im.src=window.__casebook.assetURL(h);try{await im.decode();if(im.naturalWidth!==a.width)failures.push(h)}catch(e){failures.push([h,e.message])}im.removeAttribute('src');n++}return {n,failures}}""")
                print("SEQUENTIAL", result)
            await page.screenshot(path=str(ROOT / "checks" / ("diagnose-"+name+".png")))
        await b.close()
asyncio.run(main())
