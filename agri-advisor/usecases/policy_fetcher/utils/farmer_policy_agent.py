from typing import Dict, List
from .search_tool import WebSearchTool
from .openai_client import OpenAIClient

class FarmerPolicyAgent:
    def __init__(self, search_tool: WebSearchTool, openai_client: OpenAIClient):
        self.search_tool = search_tool
        self.openai_client = openai_client
    
    async def search_policies(self, farmer_details: Dict) -> List[Dict]:
        """
        Search for relevant policies based on farmer details
        """
        try:
            # Create search queries based on farmer details
            queries = self._generate_search_queries(farmer_details)
            all_results = []
            
            for query in queries:
                print(f"Searching for: {query}")
                results = await self.search_tool._arun(query=query, k=3)
                
                if isinstance(results, str):
                    import ast
                    results = ast.literal_eval(results)
                
                if isinstance(results, list):
                    all_results.extend(results)
                else:
                    all_results.append(results)
            
            # Filter and deduplicate results
            filtered_results = self._filter_relevant_results(all_results, farmer_details)
            return filtered_results
            
        except Exception as e:
            print(f"Error in search_policies: {str(e)}")
            return []
    
    async def process_policies(self, policies_data: List[Dict], farmer_details: Dict) -> Dict:
        """
        Process and structure policy data using OpenAI
        """
        try:
            # Use OpenAI to summarize and structure the data
            structured_output = await self.openai_client.summarize_policies(
                policies_data, farmer_details
            )
            
            # Add source information
            structured_output["sources"] = [
                {
                    "title": policy.get("Title", ""),
                    "url": policy.get("Link", ""),
                    "snippet": policy.get("Snippet", "")[:200] + "..."
                }
                for policy in policies_data[:5]  # Top 5 sources
            ]
            
            return structured_output
            
        except Exception as e:
            print(f"Error in process_policies: {str(e)}")
            return {"error": str(e)}
    
    def _generate_search_queries(self, farmer_details: Dict) -> List[str]:
        """
        Generate relevant search queries based on farmer details
        """
        location = farmer_details.get('location', '').lower()
        crop_type = farmer_details.get('crop_type', '').lower()
        category = farmer_details.get('farmer_category', '').lower()
        farm_size = farmer_details.get('farm_size', '').lower()
        
        queries = [
            f"farmer schemes policies {location} {crop_type} 2024",
            f"{category} farmer government schemes {location}",
            f"agricultural subsidies {crop_type} {location}",
            f"PM-KISAN scheme eligibility {location}",
            f"crop insurance schemes {location} {crop_type}",
            f"farming loans subsidies {category} {location}"
        ]
        
        # Add specific queries based on farm size
        if any(word in farm_size for word in ['small', 'marginal', 'below']):
            queries.append(f"small farmer schemes {location} government")
            queries.append(f"marginal farmer benefits {location}")
        
        return queries[:4]  # Limit to 4 queries to avoid too many API calls
    
    def _filter_relevant_results(self, results: List[Dict], farmer_details: Dict) -> List[Dict]:
        """
        Filter results to keep only relevant policy information
        """
        relevant_results = []
        seen_urls = set()
        
        for result in results:
            if not isinstance(result, dict):
                continue
                
            url = result.get("Link", "")
            title = result.get("Title", "").lower()
            content = result.get("Content", "").lower()
            
            # Skip duplicates
            if url in seen_urls:
                continue
            
            # Check relevance
            if self._is_relevant(title, content, farmer_details):
                seen_urls.add(url)
                relevant_results.append(result)
        
        return relevant_results[:8]  # Limit to top 8 results
    
    def _is_relevant(self, title: str, content: str, farmer_details: Dict) -> bool:
        """
        Check if the search result is relevant to farming policies
        """
        # Keywords that indicate farming/agricultural content
        farming_keywords = [
            'farmer', 'agriculture', 'crop', 'farming', 'agricultural',
            'scheme', 'subsidy', 'policy', 'government', 'pm-kisan',
            'insurance', 'loan', 'msp', 'procurement', 'kisan'
        ]
        
        # Location keywords
        location = farmer_details.get('location', '').lower()
        location_words = location.split()
        
        text_to_check = f"{title} {content}".lower()
        
        # Check for farming keywords
        has_farming_keyword = any(keyword in text_to_check for keyword in farming_keywords)
        
        # Check for location relevance (if location is provided)
        has_location_relevance = True
        if location_words:
            has_location_relevance = any(word in text_to_check for word in location_words)
        
        return has_farming_keyword and (has_location_relevance or 'india' in text_to_check)