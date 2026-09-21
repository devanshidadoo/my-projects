"""
OpenAI client for policy analysis
"""
import os
import sys
import json
import asyncio
from typing import Dict, Any, List
from openai import AsyncOpenAI
# Add the parent directory (project root) to Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)

from config import Config as config

class OpenAIClient:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=getattr(config, "OPENAI_API_KEY", None))
        self.model = "gpt-4o"  # Cost-effective model
    
    async def analyze_policies(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze policies using OpenAI and return structured data
        """
        try:
            system_prompt = """
            You are an expert agricultural policy analyst specializing in Indian government schemes for farmers.
            Analyze the provided farmer profile and policy information to give practical, actionable advice.
            
            Return your response as valid JSON with the following structure:
            {
                "relevant_schemes": [
                    {
                        "scheme_name": "Name of scheme",
                        "eligibility": "Eligibility criteria",
                        "benefits": "Benefits and subsidies",
                        "documents_required": ["doc1", "doc2"],
                        "application_process": "How to apply",
                        "relevance_score": 9.5,
                        "deadline": "Application deadline if any"
                    }
                ],
                "action_plan": [
                    "Step 1: Action item",
                    "Step 2: Action item"
                ],
                "benefits_summary": {
                    "total_potential_benefits": "Estimated total benefits",
                    "immediate_benefits": "Benefits available immediately",
                    "long_term_benefits": "Long-term advantages"
                },
                "recommendations": [
                    "Recommendation 1",
                    "Recommendation 2"
                ]
            }
            
            Focus on:
            - Most relevant and applicable schemes
            - Clear eligibility criteria
            - Practical application steps
            - Required documentation
            - Timeline for benefits
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Fallback: extract JSON from text if wrapped in markdown
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_content = content[json_start:json_end].strip()
                    return json.loads(json_content)
                else:
                    # Return structured fallback
                    return {
                        "relevant_schemes": [],
                        "action_plan": ["Contact local agriculture officer for detailed information"],
                        "benefits_summary": {"message": "Analysis completed but formatting issues occurred"},
                        "raw_response": content
                    }
                    
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            return {
                "error": f"OpenAI analysis failed: {str(e)}",
                "relevant_schemes": [],
                "action_plan": ["Contact local agriculture officer for detailed information"],
                "benefits_summary": {"message": "OpenAI analysis failed"},
                "raw_response": str(e)
            }
    
    async def summarize_schemes(self, schemes_data: list) -> str:
        """
        Create a concise summary of multiple schemes
        """
        try:
            prompt = f"""
            Summarize the following government schemes for farmers in 2-3 paragraphs:
            
            {json.dumps(schemes_data, indent=2)}
            
            Focus on:
            - Key benefits available
            - Common eligibility criteria
            - Application process overview
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a policy summarization expert. Provide clear, concise summaries."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Summary generation failed: {str(e)}"

    async def summarize_policies(self, policies_data: List[Dict], farmer_details: Dict) -> Dict[str, Any]:
        """
        Summarize and structure policy data for a specific farmer
        """
        try:
            # Create a comprehensive prompt for policy summarization
            policies_text = "\n\n".join([
                f"Policy {i+1}:\nTitle: {p.get('Title', '')}\nContent: {p.get('Content', p.get('Snippet', ''))}"
                for i, p in enumerate(policies_data[:10])  # Limit to top 10 policies
            ])
            
            farmer_info = f"""
            Farmer Profile:
            - Location: {farmer_details.get('location', 'Unknown')}
            - Farm Size: {farmer_details.get('farm_size', 'Unknown')}
            - Crops: {farmer_details.get('crop_type', 'Unknown')}
            - Category: {farmer_details.get('farmer_category', 'Unknown')}
            """
            
            prompt = f"""
            {farmer_info}
            
            Available Policies and Schemes:
            {policies_text}
            
            Please analyze these policies and provide a structured summary with:
            1. Top 3 most relevant schemes for this farmer
            2. Key eligibility criteria
            3. Main benefits and subsidies
            4. Application process overview
            5. Action items for the farmer
            
            Return as JSON with clear sections.
            """
            
            system_prompt = """
            You are an expert agricultural policy analyst. Provide clear, actionable summaries 
            of government schemes for farmers. Focus on practical information and next steps.
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Fallback: extract JSON from text if wrapped in markdown
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_content = content[json_start:json_end].strip()
                    return json.loads(json_content)
                else:
                    # Return structured fallback
                    return {
                        "relevant_schemes": [],
                        "action_plan": ["Contact local agriculture officer for detailed information"],
                        "benefits_summary": {"message": "Analysis completed but formatting issues occurred"},
                        "raw_response": content
                    }
                    
        except Exception as e:
            print(f"Policy summarization error: {str(e)}")
            return {
                "error": f"Policy summarization failed: {str(e)}",
                "relevant_schemes": [],
                "action_plan": ["Contact local agriculture officer for detailed information"],
                "benefits_summary": {"message": "Policy summarization failed"},
                "raw_response": str(e)
            }

# Test function
async def test_openai_client():
    """Test the OpenAI client"""
    client = OpenAIClient()
    
    test_prompt = """
    Farmer Profile:
    - Name: Test Farmer
    - Location: Punjab
    - Farm Size: 5 acres
    - Crops: wheat, rice
    - Farming Type: conventional
    - Annual Income: ₹300,000
    - Land Ownership: owned
    
    Available Policies:
    Policy 1: PM-KISAN scheme provides ₹6000 per year to small farmers
    Policy 2: Crop insurance scheme covers weather-related losses
    """
    
    result = await client.analyze_policies(test_prompt)
    print("Test Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(test_openai_client())