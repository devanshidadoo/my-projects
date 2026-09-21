from typing import Type, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import asyncio
import json

import sys
import os

# Add the parent directory (project root) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from usecases.policy_fetcher.main import FarmerPolicyAgent
from usecases.policy_fetcher.farmer_details import FarmerDetails

# Add PrivateAttr import for pydantic v2
from pydantic import PrivateAttr

class PolicyFetcherInput(BaseModel):
    name: str = Field(..., description="Farmer's name")
    location: str = Field(..., description="Farmer's location (state/district)")
    farm_size_acres: float = Field(..., gt=0, description="Farm size in acres")
    crop_types: List[str] = Field(..., min_items=1, description="List of crops grown")
    farming_type: str = Field(..., description="Type of farming (organic, conventional, mixed)")
    annual_income: float = Field(..., ge=0, description="Annual income in INR")
    land_ownership: str = Field(..., description="Land ownership type (owned, leased, sharecropped)")

class PolicyFetcherTool(BaseTool):
    name: str = "PolicyFetcherTool"
    description: str = (
        "Fetch and analyze government policies and schemes for farmers based on their details. "
        "Returns relevant schemes, action plans, and benefits summary tailored to the farmer's profile. "
        "Inputs: farmer name, location, farm size, crop types, farming type, annual income, and land ownership. "
        "Returns structured policy recommendations with eligibility criteria, benefits, and application steps."
    )
    args_schema: Type[PolicyFetcherInput] = PolicyFetcherInput

    # Use private attrs to store non-pydantic fields
    _agent: Optional[FarmerPolicyAgent] = PrivateAttr(default=None)
    _FarmerDetails: Type[FarmerDetails] = PrivateAttr(default=FarmerDetails)
    _init_error: str = PrivateAttr(default="")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            # Check if required environment variables are set
            required_env_vars = ["OPENAI_API_KEY"]
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            
            if missing_vars:
                self._init_error = f"Missing required environment variables: {', '.join(missing_vars)}"
                print(f"Warning: {self._init_error}")
            else:
                print("PolicyFetcherTool initialized successfully")
                
        except Exception as e:
            self._init_error = f"Could not initialize policy fetcher tool: {e}"
            print(f"Warning: {self._init_error}")

    def _run(self, **kwargs) -> Any:
        """Synchronous version - not recommended, use async version instead."""
        try:
            # For sync calls, we'll run the async version in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._arun(**kwargs))
                return result
            finally:
                loop.close()
        except Exception as e:
            error_msg = str(e)
            return {"error": f"❌ Error during policy fetching: {error_msg}"}

    async def _arun(self, **kwargs) -> Any:
        """Asynchronously fetch and analyze farmer policies."""
        try:
            if self._init_error:
                return {"error": f"❌ PolicyFetcherTool initialization failed: {self._init_error}"}

            # Extract parameters
            name = kwargs.get("name")
            location = kwargs.get("location")
            farm_size_acres = kwargs.get("farm_size_acres")
            crop_types = kwargs.get("crop_types")
            farming_type = kwargs.get("farming_type")
            annual_income = kwargs.get("annual_income")
            land_ownership = kwargs.get("land_ownership")

            # Validate required parameters
            if not all([name, location, farm_size_acres, crop_types, farming_type, annual_income, land_ownership]):
                return {"error": "❌ Missing required parameters. All fields are mandatory."}

            # Create farmer details
            farmer_details = self._FarmerDetails(
                name=name,
                location=location,
                farm_size_acres=farm_size_acres,
                crop_types=crop_types,
                farming_type=farming_type,
                annual_income=annual_income,
                land_ownership=land_ownership
            )

            # Initialize agent and get policies
            agent = FarmerPolicyAgent()
            result = await agent.get_farmer_policies(farmer_details)

            # Just return the result as is (raw output from PolicyAnalyzer)
            return result

        except Exception as e:
            error_msg = str(e)
            return {"error": f"❌ Exception during policy fetching: {error_msg}"}

if __name__ == "__main__":
    import asyncio
    import json

    async def test_policy_fetcher():
        tool = PolicyFetcherTool()
        
        # Example farmer details
        test_params = {
            "name": "Rajesh Kumar",
            "location": "Punjab",
            "farm_size_acres": 5.0,
            "crop_types": ["wheat"],
            "farming_type": "conventional",
            "annual_income": 150000.0,
            "land_ownership": "owned"
        }
        
        print(f"Testing PolicyFetcherTool with farmer: {test_params['name']}")
        print("Input parameters:")
        print(json.dumps(test_params, indent=2, ensure_ascii=False))
        print("\n" + "="*50)
        
        results = await tool._arun(**test_params)
        print("Results:")
        print(json.dumps(results, indent=2, ensure_ascii=False))

    asyncio.run(test_policy_fetcher())
