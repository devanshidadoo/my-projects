"""
FastAPI endpoint for Farmer Policy Agent
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import uvicorn
from main import FarmerPolicyAgent, FarmerDetails

app = FastAPI(
    title="Farmer Policy Agent API",
    description="API to fetch and analyze government policies and schemes for farmers",
    version="1.0.0"
)

class FarmerDetailsRequest(BaseModel):
    name: str = Field(..., description="Farmer's name")
    location: str = Field(..., description="Farmer's location (state/district)")
    farm_size_acres: float = Field(..., gt=0, description="Farm size in acres")
    crop_types: List[str] = Field(..., min_items=1, description="List of crops grown")
    farming_type: str = Field(..., description="Type of farming (organic, conventional, mixed)")
    annual_income: float = Field(..., ge=0, description="Annual income in INR")
    land_ownership: str = Field(..., description="Land ownership type (owned, leased, sharecropped)")

class PolicyResponse(BaseModel):
    success: bool
    farmer_name: str
    location: str
    relevant_schemes: Optional[List[Dict[str, Any]]] = None
    action_plan: Optional[List[str]] = None
    benefits_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Initialize the agent
agent = FarmerPolicyAgent()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Farmer Policy Agent API is running",
        "version": "1.0.0",
        "endpoints": {
            "get_policies": "/farmer/policies",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "farmer-policy-agent"}

@app.post("/farmer/policies", response_model=PolicyResponse)
async def get_farmer_policies(farmer_request: FarmerDetailsRequest):
    """
    Get relevant policies and schemes for a farmer
    """
    try:
        # Convert request to FarmerDetails
        farmer_details = FarmerDetails(
            name=farmer_request.name,
            location=farmer_request.location,
            farm_size_acres=farmer_request.farm_size_acres,
            crop_types=farmer_request.crop_types,
            farming_type=farmer_request.farming_type,
            annual_income=farmer_request.annual_income,
            land_ownership=farmer_request.land_ownership
        )
        
        # Get policies using the agent
        result = await agent.get_farmer_policies(farmer_details)
        
        if "error" in result:
            return PolicyResponse(
                success=False,
                farmer_name=farmer_request.name,
                location=farmer_request.location,
                error=result["error"]
            )
        
        return PolicyResponse(
            success=True,
            farmer_name=farmer_request.name,
            location=farmer_request.location,
            relevant_schemes=result.get("relevant_schemes", []),
            action_plan=result.get("action_plan", []),
            benefits_summary=result.get("benefits_summary", {}),
            error=None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/farmer/policies/quick")
async def get_quick_farmer_policies(farmer_request: FarmerDetailsRequest):
    """
    Get a quick summary of relevant policies (lighter version)
    """
    try:
        farmer_details = FarmerDetails(
            name=farmer_request.name,
            location=farmer_request.location,
            farm_size_acres=farmer_request.farm_size_acres,
            crop_types=farmer_request.crop_types[:2],  # Limit crops for quick search
            farming_type=farmer_request.farming_type,
            annual_income=farmer_request.annual_income,
            land_ownership=farmer_request.land_ownership
        )
        
        # Use a simplified version of the agent (you could create a quick_search method)
        result = await agent.get_farmer_policies(farmer_details)
        
        # Return simplified response
        return {
            "success": True,
            "farmer_name": farmer_request.name,
            "location": farmer_request.location,
            "top_schemes": result.get("relevant_schemes", [])[:3],  # Top 3 only
            "quick_benefits": result.get("benefits_summary", {})
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in quick search: {str(e)}"
        )

@app.get("/schemes/popular/{location}")
async def get_popular_schemes(location: str):
    """
    Get popular schemes for a specific location
    """
    try:
        # Create a generic search for popular schemes
        search_tool = agent.search_tool
        query = f"popular government schemes farmers {location} 2024"
        
        results = await search_tool._arun(query=query, k=5)
        if isinstance(results, str):
            results = eval(results)
        
        return {
            "location": location,
            "popular_schemes": [
                {
                    "title": r.get("Title", ""),
                    "summary": r.get("Snippet", "")[:200] + "..." if len(r.get("Snippet", "")) > 200 else r.get("Snippet", ""),
                    "link": r.get("Link", "")
                }
                for r in results[:5]
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching popular schemes: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )