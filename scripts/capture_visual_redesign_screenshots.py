"""Script to capture dark-theme screenshots of the visual research dashboard redesign.
Saves screenshots directly to the artifact directory.
"""

from pathlib import Path
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = Path(r"C:\Users\saive\.gemini\antigravity\brain\f52e2579-9f13-4e95-9437-36e808671a7b")

def run():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # 1. Desktop 1366x768
        context = browser.new_context(
            viewport={"width": 1366, "height": 768},
            device_scale_factor=1,
            color_scheme="dark",
        )
        page = context.new_page()

        # 1. Overview Page
        print("Capturing Overview Page...")
        page.goto("http://localhost:5173/", wait_until="networkidle")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(ARTIFACT_DIR / "visual_01_overview_1366x768.png"), full_page=False)

        # 2. Experiments Page
        print("Capturing Experiments Page...")
        page.goto("http://localhost:5173/experiments", wait_until="networkidle")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(ARTIFACT_DIR / "visual_02_experiments_1366x768.png"), full_page=False)

        # 3. Experiments Page with Guided Definition Form
        print("Capturing Experiments Guided Definition Form...")
        def_btn = page.locator("button:has-text('+ Define Experiment')")
        if def_btn.count() > 0:
            def_btn.click()
            page.wait_for_timeout(800)
        page.screenshot(path=str(ARTIFACT_DIR / "visual_03_experiments_form_1366x768.png"), full_page=False)

        # 4. Results Page (Top: 2-Card Visual Dashboard)
        print("Capturing Results Page (Visual 2-Card Dashboard)...")
        page.goto("http://localhost:5173/results", wait_until="networkidle")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(ARTIFACT_DIR / "visual_04_results_1366x768.png"), full_page=False)

        # 5. Results Page with Case Inspected in Drawer (Scrolled down & accordions opened)
        print("Capturing Results Page with Focused Case Drawer...")
        # Open accordions for progressive disclosure visibility
        summaries = page.locator("details summary")
        count = summaries.count()
        for i in range(min(count, 3)):
            try:
                summaries.nth(i).click()
            except Exception:
                pass
        page.wait_for_timeout(500)
        # Scroll down to the forensic investigation section
        forensic_el = page.locator("text=Forensic Investigation").first
        if forensic_el.count() > 0:
            forensic_el.scroll_into_view_if_needed()
            page.wait_for_timeout(600)
        page.screenshot(path=str(ARTIFACT_DIR / "visual_05_results_drawer_1366x768.png"), full_page=False)

        # 6. Analysis Reliability Page
        print("Capturing Analysis Reliability Page...")
        page.goto("http://localhost:5173/analysis?tab=reliability", wait_until="networkidle")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(ARTIFACT_DIR / "visual_06_analysis_reliability_1366x768.png"), full_page=False)

        context.close()

        # 2. Tablet / Mobile 768x1024
        print("Capturing Tablet/Mobile responsive views...")
        mobile_context = browser.new_context(
            viewport={"width": 768, "height": 1024},
            device_scale_factor=1,
            color_scheme="dark",
        )
        mobile_page = mobile_context.new_page()

        mobile_page.goto("http://localhost:5173/", wait_until="networkidle")
        mobile_page.wait_for_timeout(1000)
        mobile_page.screenshot(path=str(ARTIFACT_DIR / "visual_07_overview_mobile_768x1024.png"), full_page=False)

        mobile_page.goto("http://localhost:5173/results", wait_until="networkidle")
        mobile_page.wait_for_timeout(1000)
        mobile_page.screenshot(path=str(ARTIFACT_DIR / "visual_08_results_mobile_768x1024.png"), full_page=False)

        mobile_context.close()
        browser.close()
        print("All screenshots successfully captured!")

if __name__ == "__main__":
    run()
