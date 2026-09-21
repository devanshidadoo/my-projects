from typing import Optional, Dict, Any, List, Tuple
import requests
import json
import os
from datetime import datetime, timedelta, UTC
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

class WeatherInput(BaseModel):
    """Input schema for the weather tool."""
    location: str = Field(..., description="The location to get weather data for")
    date_range: Optional[str] = Field(None, description="Optional date range in format 'YYYY-MM-DD:YYYY-MM-DD'")
    analysis_type: Optional[str] = Field(
        "current", 
        description="Type of analysis: 'current', 'forecast', or 'historical'"
    )
    model_config = ConfigDict(extra="allow")

class WeatherAnalysisTool(BaseTool):
    name: str = "WeatherAnalysisTool"
    description: str = (
        "A specialized tool for analyzing weather data and providing insights for Indian locations using Open-Meteo API.\n"
        "Optimized for Indian cities and regions. Input should be a location (preferably Indian city) and optionally a date range and analysis type.\n"
        "Returns weather analysis and insights for the specified parameters.\n"
        "Supports major Indian cities like Mumbai, Delhi, Bangalore, Chennai, Kolkata, Hyderabad, Pune, etc."
    )
    args_schema: type[BaseModel] = WeatherInput

    geocoding_api: str = "https://geocoding-api.open-meteo.com/v1/search"
    weather_api: str = "https://api.open-meteo.com/v1/forecast"

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def _load_indian_city_mappings(self) -> Dict[str, str]:
        """Load Indian city mappings from JSON file."""
        try:
            # Get the directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, "weather", "indian_cities.json")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("indian_city_mappings", {})
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            # Fallback to a minimal set of mappings if file loading fails
            print(f"Warning: Could not load city mappings from JSON: {e}")
            return {
                "mumbai": "Mumbai, India",
                "delhi": "Delhi, India",
                "bangalore": "Bangalore, India",
                "chennai": "Chennai, India",
                "kolkata": "Kolkata, India",
                "hyderabad": "Hyderabad, India",
                "pune": "Pune, India"
            }

    def _get_coordinates(self, location: str) -> Tuple[float, float]:
        """Get coordinates for a location using Open-Meteo Geocoding API."""
        params = {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json"
        }
        try:
            response = requests.get(self.geocoding_api, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data.get("results"):
                raise ValueError(f"Location not found: {location}")
            result = data["results"][0]
            return result["latitude"], result["longitude"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error accessing geocoding API: {str(e)}")

    def _get_weather_data(
        self, lat: float, lon: float, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> Dict:
        """Get weather data from Open-Meteo API."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join([
                "temperature_2m", "relative_humidity_2m", "precipitation_probability", 
                "wind_speed_10m", "weather_code"
            ]),
            "daily": ",".join([
                "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                "wind_speed_10m_max"
            ]),
            "timezone": "UTC",
            "current_weather": "true"
        }
        if start_date and end_date:
            params["start_date"] = start_date
            params["end_date"] = end_date
        try:
            response = requests.get(self.weather_api, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error accessing weather API: {str(e)}")

    def _weather_code_to_condition(self, code: int) -> str:
        """Convert Open-Meteo weather code to human readable condition."""
        conditions = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow fall",
            73: "Moderate snow fall",
            75: "Heavy snow fall",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail",
        }
        return conditions.get(code, "Unknown")

    def _run(
        self,
        query: str = None,
        location: str = None,
        date_range: str = None,
        analysis_type: str = "current",
        **kwargs: Any
    ) -> str:
        try:
            # If query is provided, try to extract location and analysis_type from it
            if query:
                query_lower = query.lower()
                extracted_location = None
                # Try to extract location from common patterns
                patterns = [
                    ("weather in", "weather in"),
                    ("current weather in", "current weather in"),
                    ("forecast for", "forecast for"),
                    ("temperature in", "temperature in"),
                    ("historical weather analysis for", "historical weather analysis for"),
                    ("weather for", "weather for"),
                    ("weather like in", "weather like in"),
                    (" in ", " in "),
                ]
                for key, pat in patterns:
                    if pat in query_lower:
                        extracted_location = query_lower.split(pat)[-1].strip()
                        break
                if not extracted_location:
                    words = query_lower.split()
                    if words:
                        extracted_location = words[-1].strip()
                # Clean up extracted location
                if extracted_location:
                    extracted_location = extracted_location.replace("?", "").replace(".", "").strip()
                    common_words = {
                        "the", "for", "next", "few", "days", "past", "week", "current", "weather",
                        "forecast", "historical", "analysis", "today", "like"
                    }
                    location_words = extracted_location.split()
                    while location_words and location_words[-1].lower() in common_words:
                        location_words.pop()
                    extracted_location = " ".join(location_words).strip()
                if not extracted_location:
                    return "Error: Could not extract location from query. Please provide a location name."
                location = extracted_location
                # Determine analysis type from query
                if "forecast" in query_lower:
                    analysis_type = "forecast"
                elif "historical" in query_lower or "past" in query_lower:
                    analysis_type = "historical"
            if not location:
                return "Error: No location provided. Please specify a location."
            # Map common Indian city names/aliases
            indian_city_mappings = self._load_indian_city_mappings()
            
            # Check if the location matches any Indian city mapping
            location_lower = location.lower()
            if location_lower in indian_city_mappings:
                location = indian_city_mappings[location_lower]
            
            # Get coordinates for the location
            lat, lon = self._get_coordinates(location)
            
            # Set up date range
            start_date = None
            end_date = None
            if analysis_type == "forecast":
                start_date = datetime.now(UTC).strftime("%Y-%m-%d")
                end_date = (datetime.now(UTC) + timedelta(days=7)).strftime("%Y-%m-%d")
            elif analysis_type == "historical":
                end_date = datetime.now(UTC).strftime("%Y-%m-%d")
                start_date = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%d")
            
            # Get weather data
            weather_data = self._get_weather_data(lat, lon, start_date, end_date)
            
            # Format based on analysis type
            if analysis_type == "current":
                return self._format_current_weather(weather_data, location)
            elif analysis_type == "forecast":
                return self._format_forecast_weather(weather_data, location)
            elif analysis_type == "historical":
                return self._format_historical_weather(weather_data, location)
            else:
                return f"Unsupported analysis type: {analysis_type}"
                
        except Exception as e:
            return f"Error: {str(e)}"

    async def arun(
        self,
        query: str = None,
        location: str = None,
        date_range: str = None,
        analysis_type: str = "current",
        **kwargs: Any
    ) -> str:
        """Async version of the run method for the weather tool."""
        try:
            # If query is provided, try to extract location and analysis_type from it
            if query:
                query_lower = query.lower()
                extracted_location = None
                # Try to extract location from common patterns
                patterns = [
                    ("weather in", "weather in"),
                    ("current weather in", "current weather in"),
                    ("forecast for", "forecast for"),
                    ("temperature in", "temperature in"),
                    ("historical weather analysis for", "historical weather analysis for"),
                    ("weather for", "weather for"),
                    ("weather like in", "weather like in"),
                    (" in ", " in "),
                ]
                for key, pat in patterns:
                    if pat in query_lower:
                        extracted_location = query_lower.split(pat)[-1].strip()
                        break
                if not extracted_location:
                    words = query_lower.split()
                    if words:
                        extracted_location = words[-1].strip()
                # Clean up extracted location
                if extracted_location:
                    extracted_location = extracted_location.replace("?", "").replace(".", "").strip()
                    common_words = {
                        "the", "for", "next", "few", "days", "past", "week", "current", "weather",
                        "forecast", "historical", "analysis", "today", "like"
                    }
                    location_words = extracted_location.split()
                    while location_words and location_words[-1].lower() in common_words:
                        location_words.pop()
                    extracted_location = " ".join(location_words).strip()
                if not extracted_location:
                    return "Error: Could not extract location from query. Please provide a location name."
                location = extracted_location
                # Determine analysis type from query
                if "forecast" in query_lower:
                    analysis_type = "forecast"
                elif "historical" in query_lower or "past" in query_lower:
                    analysis_type = "historical"
            if not location:
                return "Error: No location provided. Please specify a location."
            # Map common Indian city names/aliases
            indian_city_mappings = self._load_indian_city_mappings()

            # Check if the location matches any Indian city mapping
            location_lower = location.lower()
            if location_lower in indian_city_mappings:
                location = indian_city_mappings[location_lower]
            
            # Get coordinates for the location
            lat, lon = self._get_coordinates(location)
            
            # Set up date range
            start_date = None
            end_date = None
            if analysis_type == "forecast":
                start_date = datetime.now(UTC).strftime("%Y-%m-%d")
                end_date = (datetime.now(UTC) + timedelta(days=7)).strftime("%Y-%m-%d")
            elif analysis_type == "historical":
                end_date = datetime.now(UTC).strftime("%Y-%m-%d")
                start_date = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%d")
            
            # Get weather data
            weather_data = self._get_weather_data(lat, lon, start_date, end_date)
            
            # Format based on analysis type
            if analysis_type == "current":
                return self._format_current_weather(weather_data, location)
            elif analysis_type == "forecast":
                return self._format_forecast_weather(weather_data, location)
            elif analysis_type == "historical":
                return self._format_historical_weather(weather_data, location)
            else:
                return f"Unsupported analysis type: {analysis_type}"
                
        except Exception as e:
            return f"Error: {str(e)}"

    def _format_current_weather(self, data: Dict, location: str) -> str:
        """Format current weather data."""
        current = data["current_weather"]
        return f"""
            Current weather in {location}:
            Temperature: {current['temperature']}°C
            Conditions: {self._weather_code_to_condition(current['weathercode'])}
            Wind Speed: {current['windspeed']} km/h
            Last Updated: {current['time']} UTC
            """

    def _format_forecast_weather(self, data: Dict, location: str) -> str:
        """Format weather forecast data."""
        daily = data["daily"]
        forecast_text = f"Weather forecast for {location}:\n"
        
        for i in range(len(daily["time"])):
            forecast_text += f"""
            Date: {daily['time'][i]}
            Temperature: {daily['temperature_2m_min'][i]}°C to {daily['temperature_2m_max'][i]}°C
            Precipitation: {daily['precipitation_sum'][i]}mm
            Wind Speed (max): {daily['wind_speed_10m_max'][i]} km/h
            ---"""
        return forecast_text

    def _format_historical_weather(self, data: Dict, location: str) -> str:
        """Format historical weather data."""
        daily = data["daily"]
        
        # Calculate averages
        avg_temp_max = sum(daily["temperature_2m_max"]) / len(daily["temperature_2m_max"])
        avg_temp_min = sum(daily["temperature_2m_min"]) / len(daily["temperature_2m_min"])
        total_precipitation = sum(daily["precipitation_sum"])
        
        return f"""
            Historical weather analysis for {location}:
            Date Range: {daily['time'][0]} to {daily['time'][-1]}
            Average Temperature Range: {avg_temp_min:.1f}°C to {avg_temp_max:.1f}°C
            Total Precipitation: {total_precipitation:.1f}mm
            Summary: {len(daily['time'])} days analyzed
            """

if __name__ == "__main__":
    # Simple if-main test for the WeatherAnalysisTool
    tool = WeatherAnalysisTool()
    print("Testing WeatherAnalysisTool with Mumbai, current weather:")
    result = tool._run(location="Mumbai")

    print(result)