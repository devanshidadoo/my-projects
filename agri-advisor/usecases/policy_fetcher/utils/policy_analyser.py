"""
Policy analyzer for structuring and formatting output
"""
from datetime import datetime
import sys
import os
from typing import Dict, List, Any
# Add the project root to Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from farmer_details import FarmerDetails

class PolicyAnalyzer:
    def __init__(self):
        self.output_template = {
            "farmer_profile": {},
            "analysis_date": "",
            "relevant_schemes": [],
            "action_plan": [],
            "benefits_summary": {},
            "recommendations": [],
            "contact_information": [],
            "metadata": {}
        }
    
    def structure_output(self, analyzed_policies: Dict, farmer_details: FarmerDetails) -> Dict[str, Any]:
        """
        Structure the analyzed policies into a comprehensive output
        """
        structured_output = self.output_template.copy()
        
        # Fill farmer profile
        structured_output["farmer_profile"] = {
            "name": farmer_details.name,
            "location": farmer_details.location,
            "farm_size_acres": farmer_details.farm_size_acres,
            "crops": farmer_details.crop_types,
            "farming_type": farmer_details.farming_type,
            "annual_income": farmer_details.annual_income,
            "land_ownership": farmer_details.land_ownership
        }
        
        structured_output["analysis_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Process analyzed policies
        if "error" in analyzed_policies:
            structured_output["error"] = analyzed_policies["error"]
            structured_output["success"] = False
            return structured_output
        
        # Extract relevant schemes
        schemes = analyzed_policies.get("relevant_schemes", [])
        structured_output["relevant_schemes"] = self._format_schemes(schemes)
        
        # Extract action plan
        action_plan = analyzed_policies.get("action_plan", [])
        structured_output["action_plan"] = self._format_action_plan(action_plan, farmer_details)
        
        # Extract benefits summary
        benefits = analyzed_policies.get("benefits_summary", {})
        structured_output["benefits_summary"] = self._format_benefits_summary(benefits)
        
        # Add recommendations
        recommendations = analyzed_policies.get("recommendations", [])
        structured_output["recommendations"] = self._add_general_recommendations(recommendations, farmer_details)
        
        # Add contact information
        structured_output["contact_information"] = self._get_contact_information(farmer_details.location)
        
        # Add metadata
        structured_output["metadata"] = {
            "total_schemes_analyzed": len(schemes),
            "analysis_method": "AI-powered policy analysis",
            "data_sources": ["Government websites", "Official policy documents"],
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "confidence_level": self._calculate_confidence_level(schemes)
        }
        
        structured_output["success"] = True
        return structured_output
    
    def _format_schemes(self, schemes: List[Dict]) -> List[Dict]:
        """Format and enhance scheme information"""
        formatted_schemes = []
        
        for scheme in schemes:
            formatted_scheme = {
                "scheme_name": scheme.get("scheme_name", "Unknown Scheme"),
                "description": self._generate_description(scheme),
                "eligibility_criteria": self._format_eligibility(scheme.get("eligibility", "")),
                "benefits": self._format_benefits(scheme.get("benefits", "")),
                "required_documents": scheme.get("documents_required", []),
                "application_process": scheme.get("application_process", "Contact local agriculture office"),
                "deadline": scheme.get("deadline", "No specific deadline"),
                "relevance_score": scheme.get("relevance_score", 5.0),
                "priority": self._determine_priority(scheme),
                "estimated_benefit_amount": self._extract_benefit_amount(scheme.get("benefits", ""))
            }
            formatted_schemes.append(formatted_scheme)
        
        # Sort by relevance and priority
        formatted_schemes.sort(key=lambda x: (x["priority"], x["relevance_score"]), reverse=True)
        
        return formatted_schemes
    
    def _format_action_plan(self, action_plan: List[str], farmer_details: FarmerDetails) -> List[Dict]:
        """Format action plan with timeline and priority"""
        formatted_plan = []
        
        for i, action in enumerate(action_plan):
            formatted_action = {
                "step": i + 1,
                "action": action,
                "timeline": self._estimate_timeline(action),
                "priority": "High" if i < 3 else "Medium",
                "department": self._identify_department(action),
                "estimated_cost": self._estimate_cost(action)
            }
            formatted_plan.append(formatted_action)
        
        # Add farmer-specific actions
        formatted_plan.extend(self._add_farmer_specific_actions(farmer_details))
        
        return formatted_plan
    
    def _format_benefits_summary(self, benefits: Dict) -> Dict[str, Any]:
        """Format benefits summary with calculations"""
        formatted_benefits = {
            "immediate_benefits": {
                "description": benefits.get("immediate_benefits", "Apply for eligible schemes"),
                "estimated_amount": "₹0 - ₹50,000",
                "timeline": "1-3 months"
            },
            "annual_benefits": {
                "description": benefits.get("total_potential_benefits", "Annual scheme benefits"),
                "estimated_amount": "₹6,000 - ₹2,00,000",
                "schemes_count": "3-7 schemes"
            },
            "long_term_benefits": {
                "description": benefits.get("long_term_benefits", "Sustainable farming support"),
                "impact": "Improved income stability and risk management"
            },
            "total_estimated_annual_benefit": self._calculate_total_benefits(benefits)
        }
        
        return formatted_benefits
    
    def _add_general_recommendations(self, recommendations: List[str], farmer_details: FarmerDetails) -> List[Dict]:
        """Add general recommendations with context"""
        formatted_recommendations = []
        
        # Process AI recommendations
        for rec in recommendations:
            formatted_recommendations.append({
                "recommendation": rec,
                "category": self._categorize_recommendation(rec),
                "priority": "Medium"
            })
        
        # Add farmer-specific recommendations
        if farmer_details.farm_size_acres <= 2:
            formatted_recommendations.append({
                "recommendation": "Consider joining a Farmer Producer Organization (FPO) for better market access",
                "category": "Market Access",
                "priority": "High"
            })
        
        if farmer_details.annual_income < 200000:
            formatted_recommendations.append({
                "recommendation": "Prioritize direct benefit transfer schemes like PM-KISAN",
                "category": "Financial Support",
                "priority": "High"
            })
        
        if "organic" in farmer_details.farming_type.lower():
            formatted_recommendations.append({
                "recommendation": "Explore organic certification and premium market opportunities",
                "category": "Value Addition",
                "priority": "Medium"
            })
        
        return formatted_recommendations
    
    def _get_contact_information(self, location: str) -> List[Dict]:
        """Get relevant contact information"""
        contacts = [
            {
                "office": "District Collector Office",
                "purpose": "General scheme information and applications",
                "contact_method": f"Visit local district collector office in {location}"
            },
            {
                "office": "Agriculture Extension Office",
                "purpose": "Technical support and scheme guidance",
                "contact_method": "Contact local Krishi Vigyan Kendra (KVK)"
            },
            {
                "office": "PM-KISAN Helpline",
                "purpose": "PM-KISAN scheme queries",
                "contact_method": "Call 155261 or visit pmkisan.gov.in"
            },
            {
                "office": "Kisan Call Centre",
                "purpose": "24/7 farming queries and support",
                "contact_method": "Call 1800-180-1551"
            }
        ]
        
        return contacts
    
    # Helper methods
    def _generate_description(self, scheme: Dict) -> str:
        """Generate a brief description of the scheme"""
        name = scheme.get("scheme_name", "")
        benefits = scheme.get("benefits", "")
        
        if len(benefits) > 100:
            return benefits[:100] + "..."
        return benefits or f"Government scheme: {name}"
    
    def _format_eligibility(self, eligibility: str) -> List[str]:
        """Format eligibility criteria into a list"""
        if not eligibility:
            return ["Contact local agriculture office for eligibility details"]
        
        # Split by common separators
        criteria = [e.strip() for e in eligibility.replace("•", ",").split(",") if e.strip()]
        return criteria[:5]  # Limit to top 5 criteria
    
    def _format_benefits(self, benefits: str) -> Dict[str, str]:
        """Format benefits into structured data"""
        return {
            "financial": self._extract_financial_benefits(benefits),
            "non_financial": self._extract_non_financial_benefits(benefits),
            "summary": benefits[:200] + "..." if len(benefits) > 200 else benefits
        }
    
    def _determine_priority(self, scheme: Dict) -> int:
        """Determine scheme priority (1-5, 5 being highest)"""
        score = scheme.get("relevance_score", 5.0)
        name = scheme.get("scheme_name", "").lower()
        
        # High priority schemes
        if any(keyword in name for keyword in ["pm kisan", "crop insurance", "kisan credit"]):
            return 5
        
        if score > 8.0:
            return 5
        elif score > 6.0:
            return 4
        elif score > 4.0:
            return 3
        else:
            return 2
    
    def _extract_benefit_amount(self, benefits: str) -> str:
        """Extract monetary benefit amounts from text"""
        import re
        
        # Look for patterns like ₹1000, Rs. 1000, etc.
        patterns = [
            r'₹\s*[\d,]+',
            r'Rs\.?\s*[\d,]+',
            r'INR\s*[\d,]+',
            r'rupees?\s*[\d,]+'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, benefits, re.IGNORECASE)
            if match:
                return match.group()
        
        return "Amount varies"
    
    def _estimate_timeline(self, action: str) -> str:
        """Estimate timeline for an action"""
        action_lower = action.lower()
        
        if any(word in action_lower for word in ["document", "collect", "prepare"]):
            return "1-2 weeks"
        elif any(word in action_lower for word in ["apply", "submit"]):
            return "1 week"
        elif any(word in action_lower for word in ["visit", "contact"]):
            return "Immediate"
        else:
            return "2-4 weeks"
    
    def _identify_department(self, action: str) -> str:
        """Identify relevant department for an action"""
        action_lower = action.lower()
        
        if any(word in action_lower for word in ["bank", "credit", "loan"]):
            return "Banking/Financial Institution"
        elif any(word in action_lower for word in ["agriculture", "krishi"]):
            return "Agriculture Department"
        elif any(word in action_lower for word in ["collector", "tehsil"]):
            return "Revenue Department"
        else:
            return "Local Administration"
    
    def _estimate_cost(self, action: str) -> str:
        """Estimate cost for an action"""
        action_lower = action.lower()
        
        if any(word in action_lower for word in ["document", "certificate"]):
            return "₹100 - ₹500"
        elif "travel" in action_lower or "visit" in action_lower:
            return "₹50 - ₹200"
        else:
            return "Free"
    
    def _add_farmer_specific_actions(self, farmer_details: FarmerDetails) -> List[Dict]:
        """Add farmer-specific action items"""
        specific_actions = []
        
        if farmer_details.farm_size_acres <= 2:
            specific_actions.append({
                "step": 100,  # Will be reordered later
                "action": "Get small/marginal farmer certificate from local revenue office",
                "timeline": "1-2 weeks",
                "priority": "High",
                "department": "Revenue Department",
                "estimated_cost": "₹100 - ₹200"
            })
        
        return specific_actions
    
    def _calculate_total_benefits(self, benefits: Dict) -> str:
        """Calculate estimated total annual benefits"""
        # This is a simplified calculation
        # In reality, you'd parse actual benefit amounts
        return "₹15,000 - ₹1,50,000 per year"
    
    def _categorize_recommendation(self, recommendation: str) -> str:
        """Categorize a recommendation"""
        rec_lower = recommendation.lower()
        
        if any(word in rec_lower for word in ["money", "income", "financial", "loan"]):
            return "Financial Support"
        elif any(word in rec_lower for word in ["crop", "seed", "farming"]):
            return "Agricultural Practice"
        elif any(word in rec_lower for word in ["market", "sell", "price"]):
            return "Market Access"
        elif any(word in rec_lower for word in ["training", "skill", "knowledge"]):
            return "Capacity Building"
        else:
            return "General"
    
    def _extract_financial_benefits(self, benefits: str) -> str:
        """Extract financial benefits from text"""
        import re
        
        financial_keywords = ["₹", "rupees", "money", "amount", "subsidy", "loan", "credit"]
        
        if any(keyword in benefits.lower() for keyword in financial_keywords):
            # Extract sentence containing financial info
            sentences = benefits.split('.')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in financial_keywords):
                    return sentence.strip()
        
        return "Financial support available"
    
    def _extract_non_financial_benefits(self, benefits: str) -> str:
        """Extract non-financial benefits from text"""
        non_financial_keywords = ["training", "support", "guidance", "technical", "assistance"]
        
        sentences = benefits.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in non_financial_keywords):
                return sentence.strip()
        
        return "Technical and advisory support"
    
    def _calculate_confidence_level(self, schemes: List[Dict]) -> str:
        """Calculate confidence level of analysis"""
        if len(schemes) >= 5:
            return "High"
        elif len(schemes) >= 3:
            return "Medium"
        else:
            return "Low"

# Test function
def test_policy_analyzer():
    """Test the policy analyzer"""
    # Import here to avoid circular imports
    from usecases.policy_fetcher.farmer_details import FarmerDetails
    
    analyzer = PolicyAnalyzer()
    
    # Test data
    farmer = FarmerDetails(
        name="Test Farmer",
        location="Punjab",
        farm_size_acres=3.0,
        crop_types=["wheat", "rice"],
        farming_type="conventional",
        annual_income=300000,
        land_ownership="owned"
    )
    
    test_analyzed_policies = {
        "relevant_schemes": [
            {
                "scheme_name": "PM-KISAN",
                "eligibility": "Small and marginal farmers",
                "benefits": "₹6000 per year direct benefit transfer",
                "documents_required": ["Aadhaar", "Land records"],
                "application_process": "Online application",
                "relevance_score": 9.0
            }
        ],
        "action_plan": [
            "Collect required documents",
            "Apply for PM-KISAN scheme online"
        ],
        "benefits_summary": {
            "immediate_benefits": "Direct cash support",
            "total_potential_benefits": "₹50,000 annually"
        },
        "recommendations": [
            "Apply for crop insurance",
            "Consider soil health card"
        ]
    }
    
    result = analyzer.structure_output(test_analyzed_policies, farmer)
    
    print("Structured Output:")
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_policy_analyzer()