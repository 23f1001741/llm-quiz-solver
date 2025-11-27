import asyncio
from playwright.async_api import async_playwright

async def main():
    print(" despertar... (waking up...)")
    
    # Start the Playwright tool
    async with async_playwright() as p:
        # Launch the browser. headless=False means "Show me the window!"
        print("Launching Browser...")
        browser = await p.chromium.launch(headless=False)
        
        # Open a new tab
        page = await browser.new_page()
        
        # Go to a website
        print("Visiting example.com...")
        await page.goto("https://example.com")
        
        # Read the title of the page
        title = await page.title()
        print(f"SUCCESS! The page title is: {title}")
        
        # Wait 3 seconds so you can see it before it closes
        await asyncio.sleep(3)
        
        await browser.close()
        print("Browser closed.")

# Run the function
asyncio.run(main())