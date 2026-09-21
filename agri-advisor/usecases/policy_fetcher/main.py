"""
Main orchestrator for the Farmer Policy Agent
"""
import asyncio
import json
import os
import sys

# Add the project root to Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
)

from typing import Dict, List, Any
from dataclasses import dataclass
from .utils.openai_client import OpenAIClient
from .utils.data_processor import DataProcessor
from .utils.policy_analyser import PolicyAnalyzer
from .farmer_details import FarmerDetails

from .utils.search_tool import WebSearchTool

class FarmerPolicyAgent:
    def __init__(self):
        self.search_tool = WebSearchTool()
        self.openai_client = OpenAIClient()
        self.data_processor = DataProcessor()
        self.policy_analyzer = PolicyAnalyzer()
    
    async def get_farmer_policies(self, farmer_details: FarmerDetails) -> Dict[str, Any]:
        """
        Main method to fetch and analyze policies for a farmer
        """
        try:
            print(f"Processing request for farmer: {farmer_details.name}")
            
            # Step 1: Generate targeted search queries
            search_queries = self._generate_search_queries(farmer_details)
            
            # Step 2: Search for policies and schemes
            raw_data = await self._search_policies(search_queries)
            
            # Step 3: Process and filter relevant data
            relevant_policies = self.data_processor.filter_relevant_policies(
                raw_data, farmer_details
            )
            
            # Step 4: Analyze policies using OpenAI
            analyzed_policies = await self._analyze_policies(relevant_policies, farmer_details)
            
            # Step 5: Structure the output
            structured_output = self.policy_analyzer.structure_output(
                analyzed_policies, farmer_details
            )
            
            return structured_output
            
        except Exception as e:
            print(f"Error processing farmer policies: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _generate_search_queries(self, farmer_details: FarmerDetails) -> List[str]:
        """Generate targeted search queries based on farmer details"""
        base_queries = [
            f"government schemes farmers {farmer_details.location} 2024 2025",
            f"agricultural policies {farmer_details.location} subsidies",
            "PM KISAN farmer benefits eligibility",
            "crop insurance schemes farmers India",
            f"{farmer_details.farming_type} farming subsidies {farmer_details.location}"
        ]
        
        # Add crop-specific queries
        for crop in farmer_details.crop_types[:2]:  # Limit to 2 main crops
            base_queries.append(f"{crop} farming schemes government benefits")
        
        # Add farm size specific queries
        if farmer_details.farm_size_acres <= 2:
            base_queries.append("small farmers schemes marginal farmers benefits")
        elif farmer_details.farm_size_acres > 10:
            base_queries.append("large farmers schemes commercial agriculture benefits")
        
        return base_queries[:6]  # Limit to 6 queries to avoid overwhelming
    
    async def _search_policies(self, queries: List[str]) -> List[Dict]:
        """Search for policies using multiple queries in parallel and show which search result has completed"""
        import asyncio

        async def search_single_query(query, idx):
            try:
                print(f"Searching ({idx+1}/{len(queries)}): {query}")
                results = await self.search_tool._arun(query=query, k=1)
                if isinstance(results, str):
                    results = eval(results)  # Convert string to list if needed
                await asyncio.sleep(1)  # Rate limiting per query
                print(f"Completed search ({idx+1}/{len(queries)}): {query}")
                return results
            except Exception as e:
                print(f"Error searching for query '{query}': {str(e)}")
                return []

        # Launch all searches in parallel, tracking which query is which
        tasks = [search_single_query(query, idx) for idx, query in enumerate(queries)]
        all_results_nested = await asyncio.gather(*tasks)
        all_results = []
        for results in all_results_nested:
            if isinstance(results, list):
                all_results.extend(results)
            elif results:  # In case a single dict is returned
                all_results.append(results)
        return all_results
    
    async def _analyze_policies(self, policies: List[Dict], farmer_details: FarmerDetails) -> Dict:
        """Analyze policies using OpenAI"""
        try:
            analysis_prompt = self._create_analysis_prompt(policies, farmer_details)
            analysis = await self.openai_client.analyze_policies(analysis_prompt)
            return analysis
        except Exception as e:
            print(f"Error analyzing policies: {str(e)}")
            return {"error": "Analysis failed", "policies": policies}
    
    def _create_analysis_prompt(self, policies: List[Dict], farmer_details: FarmerDetails) -> str:
        """Create prompt for OpenAI analysis"""
        farmer_info = f"""
        Farmer Profile:
        - Name: {farmer_details.name}
        - Location: {farmer_details.location}
        - Farm Size: {farmer_details.farm_size_acres} acres
        - Crops: {', '.join(farmer_details.crop_types)}
        - Farming Type: {farmer_details.farming_type}
        - Annual Income: ₹{farmer_details.annual_income:,.2f}
        - Land Ownership: {farmer_details.land_ownership}
        """
        
        policies_text = "\n\n".join([
            f"Policy {i+1}:\nTitle: {p.get('Title', '')}\nContent: {p.get('Content', p.get('Snippet', ''))}"
            for i, p in enumerate(policies[:10])  # Limit to top 10 policies
        ])
        
        return f"""
        {farmer_info}
        
        Available Policies and Schemes:
        {policies_text}
        
        Please analyze these policies and provide:
        1. Top 5 most relevant schemes for this farmer
        2. Eligibility criteria for each scheme
        3. Benefits and subsidies available
        4. Application process and required documents
        5. Action plan for the farmer to apply
        
        Structure your response as JSON with clear sections.
        """

async def main():
    """Test the farmer policy agent"""
    agent = FarmerPolicyAgent()
    
    # Example farmer details
    farmer = FarmerDetails(
        name="Rajesh Kumar",
        location="Punjab",
        farm_size_acres=5.5,
        crop_types=["wheat", "rice", "sugarcane"],
        farming_type="conventional",
        annual_income=300000,
        land_ownership="owned"
    )
    
    print("Starting farmer policy analysis...")
    result = await agent.get_farmer_policies(farmer)
    
    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())