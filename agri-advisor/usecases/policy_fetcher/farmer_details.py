from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class FarmerDetails:
    name: str
    location: str
    farm_size_acres: float
    crop_types: List[str]
    farming_type: str  # organic, conventional, mixed
    annual_income: float
    land_ownership: str  # owned, leased, sharecropped