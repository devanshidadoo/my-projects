import difflib
from typing import List, Dict, Any, Optional

class FuzzyMatcher:
    def match(self, query: str, choices: List[Dict[str, Any]], key: str = "text") -> Optional[Dict[str, Any]]:
        names = [c[key] for c in choices]
        match = difflib.get_close_matches(query, names, n=1, cutoff=0.6)
        if match:
            for c in choices:
                if c[key] == match[0]:
                    return c
        return None

    def filter(self, query: Optional[str], values: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        if not query:
            return values
        names = [v.get(key, "") for v in values]
        match = difflib.get_close_matches(query, names, n=1, cutoff=0.6)
        if not match:
            return []
        return [v for v in values if v.get(key, "") == match[0]] 