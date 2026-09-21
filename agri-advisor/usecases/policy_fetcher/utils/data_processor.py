"""
Data processor for filtering and cleaning policy data
"""
import re
import sys
import os
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from farmer_details import FarmerDetails

class DataProcessor:
    def __init__(self):
        self.indian_states = [
            'andhra pradesh', 'assam', 'bihar', 'chhattisgarh', 'goa', 'gujarat',
            'haryana', 'himachal pradesh', 'jharkhand', 'karnataka', 'kerala',
            'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram',
            'nagaland', 'odisha', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu',
            'telangana', 'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal'
        ]
        
        self.common_crops = [
            'rice', 'wheat', 'sugarcane', 'cotton', 'jute', 'tea', 'coffee',
            'spices', 'oilseeds', 'pulses', 'maize', 'barley', 'millets',
            'fruits', 'vegetables', 'tobacco', 'coconut', 'rubber'
        ]
    
    def filter_relevant_policies(self, raw_data: List[Dict], farmer_details: FarmerDetails) -> List[Dict]:
        """
        Filter policies based on farmer details and relevance
        """
        relevant_policies = []
        
        for policy in raw_data:
            if not isinstance(policy, dict):
                continue
                
            # Skip error results
            if policy.get('Success') is False:
                continue
            
            relevance_score = self._calculate_relevance_score(policy, farmer_details)
            
            if relevance_score > 0.3:  # Minimum relevance threshold
                policy['relevance_score'] = relevance_score
                relevant_policies.append(policy)
        
        # Sort by relevance score
        relevant_policies.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Remove duplicates based on title similarity
        unique_policies = self._remove_duplicates(relevant_policies)
        
        return unique_policies[:15]  # Return top 15 most relevant
    
    def _calculate_relevance_score(self, policy: Dict, farmer_details: FarmerDetails) -> float:
        """
        Calculate relevance score for a policy based on farmer details
        """
        score = 0.0
        content = f"{policy.get('Title', '')} {policy.get('Content', '')} {policy.get('Snippet', '')}".lower()
        
        # Location relevance (high weight)
        if farmer_details.location.lower() in content:
            score += 0.3
        
        # Check for state names
        for state in self.indian_states:
            if state in farmer_details.location.lower() and state in content:
                score += 0.25
                break
        
        # Crop relevance
        crop_matches = 0
        for crop in farmer_details.crop_types:
            if crop.lower() in content:
                crop_matches += 1
        
        if crop_matches > 0:
            score += min(0.25, crop_matches * 0.1)
        
        # Farm size relevance
        if farmer_details.farm_size_acres <= 2:
            if any(term in content for term in ['small farmer', 'marginal farmer', 'small and marginal']):
                score += 0.2
        elif farmer_details.farm_size_acres > 10:
            if any(term in content for term in ['large farmer', 'commercial']):
                score += 0.15
        
        # Farming type relevance
        if farmer_details.farming_type.lower() in content:
            score += 0.1
        
        # Income-based schemes
        if farmer_details.annual_income < 200000:
            if any(term in content for term in ['bpl', 'below poverty', 'low income']):
                score += 0.15
        
        # Land ownership relevance
        if farmer_details.land_ownership.lower() in content:
            score += 0.1
        
        # Common scheme keywords (always relevant)
        scheme_keywords = [
            'pm kisan', 'pradhan mantri', 'kisan credit', 'crop insurance',
            'subsidy', 'scheme', 'yojana', 'loan', 'credit', 'benefit',
            'agriculture', 'farming', 'farmer', 'kisan'
        ]
        
        keyword_matches = sum(1 for keyword in scheme_keywords if keyword in content)
        score += min(0.2, keyword_matches * 0.03)
        
        # Recency bonus (if content suggests recent/current scheme)
        if any(year in content for year in ['2024', '2025', 'current', 'new']):
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _remove_duplicates(self, policies: List[Dict]) -> List[Dict]:
        """
        Remove duplicate policies based on title similarity
        """
        unique_policies = []
        seen_titles = set()
        
        for policy in policies:
            title = policy.get('Title', '').lower()
            
            # Simple duplicate detection
            title_words = set(re.findall(r'\w+', title))
            
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(re.findall(r'\w+', seen_title))
                
                # If more than 70% words are common, consider duplicate
                if len(title_words) > 0 and len(seen_words) > 0:
                    common_words = title_words.intersection(seen_words)
                    similarity = len(common_words) / min(len(title_words), len(seen_words))
                    
                    if similarity > 0.7:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_policies.append(policy)
                seen_titles.add(title)
        
        return unique_policies
    
    def clean_policy_content(self, policy: Dict) -> Dict:
        """
        Clean and normalize policy content
        """
        cleaned_policy = policy.copy()
        
        # Clean title
        if 'Title' in cleaned_policy:
            cleaned_policy['Title'] = re.sub(r'\s+', ' ', cleaned_policy['Title']).strip()
        
        # Clean content
        content_field = cleaned_policy.get('Content') or cleaned_policy.get('Snippet', '')
        if content_field:
            # Remove extra whitespace
            content_field = re.sub(r'\s+', ' ', content_field)
            # Remove special characters that might cause issues
            content_field = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', content_field)
            cleaned_policy['Content'] = content_field.strip()
        
        return cleaned_policy
    
    def extract_key_information(self, policies: List[Dict]) -> Dict[str, Any]:
        """
        Extract key information from policies for quick overview
        """
        info = {
            'total_policies': len(policies),
            'scheme_types': set(),
            'common_benefits': [],
            'eligibility_patterns': []
        }
        
        for policy in policies:
            content = f"{policy.get('Title', '')} {policy.get('Content', '')}".lower()
            
            # Identify scheme types
            if 'insurance' in content:
                info['scheme_types'].add('Insurance')
            if 'loan' in content or 'credit' in content:
                info['scheme_types'].add('Credit/Loan')
            if 'subsidy' in content:
                info['scheme_types'].add('Subsidy')
            if 'pension' in content:
                info['scheme_types'].add('Pension')
            if 'training' in content:
                info['scheme_types'].add('Training')
        
        info['scheme_types'] = list(info['scheme_types'])
        return info

# Test function
def test_data_processor():
    """Test the data processor"""
    # Import here to avoid circular imports
    from usecases.policy_fetcher.farmer_details import FarmerDetails
    
    processor = DataProcessor()
    
    # Test farmer
    farmer = FarmerDetails(
        name="Test Farmer",
        location="Punjab",
        farm_size_acres=3.5,
        crop_types=["wheat", "rice"],
        farming_type="conventional",
        annual_income=250000,
        land_ownership="owned"
    )
    
    # Test policies
    test_policies = [
        {
            "Title": "PM-KISAN Scheme for Small Farmers",
            "Content": "Direct benefit transfer of Rs 6000 per year to small and marginal farmers in Punjab",
            "Success": True,
            "Link": "https://example.com"
        },
        {
            "Title": "Wheat Procurement Policy",
            "Content": "Government procurement of wheat at MSP in Punjab for wheat farmers",
            "Success": True,
            "Link": "https://example2.com"
        }
    ]
    
    filtered = processor.filter_relevant_policies(test_policies, farmer)
    print("Filtered Policies:")
    for policy in filtered:
        print(f"- {policy['Title']} (Score: {policy['relevance_score']:.2f})")

if __name__ == "__main__":
    test_data_processor()