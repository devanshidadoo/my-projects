import requests
import os
import json
from typing import Optional
from usecases.crop_prices.utils.fuzzy import FuzzyMatcher
from usecases.crop_prices.utils.url_builder import AgmarknetURLBuilder
from usecases.crop_prices.utils.html_parser import AgmarknetHTMLParser

from usecases.crop_prices.config import Config as config

class CropPriceFetcher:
    def __init__(self):
        # Load commodity and state mappings
        with open(os.path.join(os.path.dirname(__file__), config.COMMODITIES), encoding="utf-8") as f:
            self.commodities = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), config.STATES), encoding="utf-8") as f:
            self.states = json.load(f)
        self.fuzzy = FuzzyMatcher()
        self.url_builder = AgmarknetURLBuilder()
        self.html_parser = AgmarknetHTMLParser()

    def fetch_prices(self, commodity: str, state: str, district: Optional[str], start_date: str, end_date: str):
        # Fuzzy match commodity and state
        commodity_obj = self.fuzzy.match(commodity, self.commodities)
        state_obj = self.fuzzy.match(state, self.states)
        if not commodity_obj:
            return {"success": False, "error": f"Commodity '{commodity}' not found."}
        if not state_obj:
            return {"success": False, "error": f"State '{state}' not found."}
        commodity_id = commodity_obj["value"]
        state_id = state_obj["value"]
        commodity_name = commodity_obj["text"]
        state_name = state_obj["text"]

        # Always fetch the whole state (district=0)
        url = self.url_builder.build(
            commodity_id, state_id, commodity_name, state_name, start_date, end_date
        )
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
        except Exception as e:
            return {"success": False, "error": f"Failed to fetch data: {e}"}
        html = response.text

        # Parse the table
        json_list = self.html_parser.parse(html)
        # If district is provided, filter results
        if district:
            json_list = self.fuzzy.filter(district, json_list, key="District_Name")
        return {"success": True, "data": json_list, "total_records": len(json_list)}

if __name__ == "__main__":
    fetcher = CropPriceFetcher()

    # Example test parameters
    commodity = "Potato"
    state = "West Bengal"
    district = None  # or "Alipurduar"
    start_date = "01-Aug-2025"
    end_date = "07-Aug-2025"
    result = fetcher.fetch_prices(commodity, state, district, start_date, end_date)
    print(json.dumps(result, ensure_ascii=False, indent=2))
