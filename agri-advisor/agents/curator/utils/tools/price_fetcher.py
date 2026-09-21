from typing import Type, Optional, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import asyncio
import json

import sys
import os

# Add the parent directory (project root) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from usecases.crop_prices.fetcher import CropPriceFetcher
from usecases.crop_prices.utils import *

# Add PrivateAttr import for pydantic v2
from pydantic import PrivateAttr

class PriceFetcherInput(BaseModel):
    commodity: str = Field(..., description="Commodity name, e.g., 'Potato'")
    state: str = Field(..., description="State name, e.g., 'West Bengal'")
    district: Optional[str] = Field(None, description="Optional district name to filter results, e.g., 'Alipurduar'")
    start_date: str = Field(..., description="Start date in format 'DD-Mon-YYYY', e.g., '01-Aug-2025'")
    end_date: str = Field(..., description="End date in format 'DD-Mon-YYYY', e.g., '07-Aug-2025'")
    analysis: str = Field("summary", description="Output format: 'summary' for market-wise summary or 'detailed' for full table")

class PriceFetcherTool(BaseTool):
    name: str = "PriceFetcherTool"
    description: str = (
        "Fetch crop price data from Agmarknet via the crop-prices usecase. "
        "Inputs: commodity, state, optional district, start_date, end_date, analysis. "
        "Analysis can be 'summary' (market-wise summary) or 'detailed' (full table format). "
        "Returns structured crop price data in an easy-to-read format with summary statistics."
    )
    args_schema: Type[PriceFetcherInput] = PriceFetcherInput

    # Use private attrs to store non-pydantic fields
    _fetcher: Optional[CropPriceFetcher] = PrivateAttr(default=None)
    _init_error: Optional[str] = PrivateAttr(default=None)

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        try:
            self._fetcher = CropPriceFetcher()
        except Exception as e:
            self._init_error = str(e)

    def _format_price_data_as_table(self, data: list, commodity: str, state: str, district: str = None) -> str:
        """Format price data as a structured table with summary statistics."""
        if not data:
            return "No price data found for the specified criteria."
        
        # Calculate summary statistics
        prices = []
        for item in data:
            try:
                min_price = float(item.get("Min_Price_Rs./Quintal", 0))
                max_price = float(item.get("Max_Price_Rs./Quintal", 0))
                modal_price = float(item.get("Modal_Price_Rs./Quintal", 0))
                if min_price > 0 and max_price > 0 and modal_price > 0:
                    prices.extend([min_price, max_price, modal_price])
            except (ValueError, TypeError):
                continue
        
        # Summary statistics
        if prices:
            avg_price = sum(prices) / len(prices)
            min_overall = min(prices)
            max_overall = max(prices)
            price_range = max_overall - min_overall
        else:
            avg_price = min_overall = max_overall = price_range = 0
        
        # Create header
        location_info = f"{commodity} prices in {state}"
        if district:
            location_info += f" ({district})"
        
        header = f"""
{'='*80}
{location_info.upper()}
{'='*80}

SUMMARY STATISTICS:
• Total Records: {len(data)}
• Average Price: ₹{avg_price:.2f}/Quintal
• Price Range: ₹{min_overall:.2f} - ₹{max_overall:.2f}/Quintal
• Overall Range: ₹{price_range:.2f}/Quintal

DETAILED PRICE DATA:
{'-'*80}
"""
        
        # Create table headers
        table_headers = [
            "Date", "Market", "Variety", "Grade", 
            "Min Price", "Max Price", "Modal Price"
        ]
        
        # Format table headers
        header_row = f"{'Date':<12} {'Market':<15} {'Variety':<12} {'Grade':<8} {'Min':<10} {'Max':<10} {'Modal':<10}"
        separator = "-" * 80
        
        # Build table rows
        table_rows = [header_row, separator]
        
        for item in data:
            date = item.get("Price_Date", "N/A")
            market = item.get("Market_Name", "N/A")[:14]  # Truncate if too long
            variety = item.get("Variety", "N/A")[:11]
            grade = item.get("Grade", "N/A")[:7]
            min_price = item.get("Min_Price_Rs./Quintal", "N/A")
            max_price = item.get("Max_Price_Rs./Quintal", "N/A")
            modal_price = item.get("Modal_Price_Rs./Quintal", "N/A")
            
            row = f"{date:<12} {market:<15} {variety:<12} {grade:<8} ₹{min_price:<9} ₹{max_price:<9} ₹{modal_price:<9}"
            table_rows.append(row)
        
        # Add footer
        footer = f"""
{'-'*80}
Total Records: {len(data)}
{'='*80}
"""
        
        return header + "\n".join(table_rows) + footer

    def _format_price_data_as_summary(self, data: list, commodity: str, state: str, district: str = None) -> str:
        """Format price data as a concise summary with key insights."""
        if not data:
            return "No price data found for the specified criteria."
        
        # Group by market and calculate averages
        market_prices = {}
        for item in data:
            market = item.get("Market_Name", "Unknown")
            try:
                min_price = float(item.get("Min_Price_Rs./Quintal", 0))
                max_price = float(item.get("Max_Price_Rs./Quintal", 0))
                modal_price = float(item.get("Modal_Price_Rs./Quintal", 0))
                
                if market not in market_prices:
                    market_prices[market] = {"min": [], "max": [], "modal": [], "dates": []}
                
                if min_price > 0 and max_price > 0 and modal_price > 0:
                    market_prices[market]["min"].append(min_price)
                    market_prices[market]["max"].append(max_price)
                    market_prices[market]["modal"].append(modal_price)
                    market_prices[market]["dates"].append(item.get("Price_Date", ""))
            except (ValueError, TypeError):
                continue
        
        # Create summary
        location_info = f"{commodity} prices in {state}"
        if district:
            location_info += f" ({district})"
        
        summary = f"""
{'='*60}
{location_info.upper()}
{'='*60}

MARKET SUMMARY:
"""
        
        for market, prices in market_prices.items():
            if prices["min"] and prices["max"] and prices["modal"]:
                avg_min = sum(prices["min"]) / len(prices["min"])
                avg_max = sum(prices["max"]) / len(prices["max"])
                avg_modal = sum(prices["modal"]) / len(prices["modal"])
                date_range = f"{prices['dates'][0]} to {prices['dates'][-1]}"
                
                summary += f"""
{market}:
  • Date Range: {date_range}
  • Average Min Price: ₹{avg_min:.2f}/Quintal
  • Average Max Price: ₹{avg_max:.2f}/Quintal
  • Average Modal Price: ₹{avg_modal:.2f}/Quintal
  • Records: {len(prices['min'])}
"""
        
        summary += f"""
{'='*60}
Total Markets: {len(market_prices)}
Total Records: {len(data)}
{'='*60}
"""
        
        return summary

    async def _arun(self, **kwargs) -> str:
        """Asynchronously fetch crop prices using the underlying fetcher."""
        if self._init_error:
            return f"PriceFetcherTool initialization failed: {self._init_error}"

        commodity = kwargs.get("commodity")
        state = kwargs.get("state")
        district = kwargs.get("district")
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")
        analysis = kwargs.get("analysis")

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                self._fetcher.fetch_prices,
                commodity,
                state,
                district,
                start_date,
                end_date,
            )
            
            # Check if the result is successful
            if isinstance(result, dict) and result.get("success"):
                data = result.get("data", [])
                total_records = result.get("total_records", 0)
                
                if analysis == "summary":
                    formatted_result = self._format_price_data_as_summary(data, commodity, state, district)
                else: # analysis == "detailed"
                    formatted_result = self._format_price_data_as_table(data, commodity, state, district)
                
                # Add a brief summary at the end
                if total_records > 0:
                    formatted_result += f"\n\n📊 QUICK SUMMARY: Found {total_records} price records for {commodity} in {state}"
                    if district:
                        formatted_result += f" ({district})"
                
                return formatted_result
            else:
                error_msg = result.get("error", "Unknown error occurred") if isinstance(result, dict) else str(result)
                return f"❌ Error fetching price data: {error_msg}"
                
        except Exception as e:
            return f"❌ Exception during price fetch: {str(e)}"

    def _run(self, **kwargs):
        raise NotImplementedError("Sync version not implemented. Use async version with _arun().") 

if __name__ == "__main__":
    import sys

    async def main():
        # Example test input
        test_input = {
            "commodity": "Rice",
            "state": "West Bengal",
            "district": "Alipurduar",
            "start_date": "01-Aug-2025",
            "end_date": "07-Aug-2025",
            "analysis": "detailed" # Test with summary format
        }
        tool = PriceFetcherTool()
        print("Testing PriceFetcherTool with input:")
        print(json.dumps(test_input, indent=2))
        result = await tool._arun(**test_input)
        print("\nResult:")
        print(result)

    asyncio.run(main())