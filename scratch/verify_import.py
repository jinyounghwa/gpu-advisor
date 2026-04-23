import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

try:
    from crawlers.danawa_crawler import DanawaCrawler
    print("Success: DanawaCrawler imported successfully")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
