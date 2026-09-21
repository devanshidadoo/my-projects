# Farmer Policy Agent

An AI-powered agent that fetches, analyzes, and structures information about government policies and schemes for farmers based on their specific details and requirements.

## Features

- **Personalized Policy Search**: Fetches relevant government schemes based on farmer details
- **AI-Powered Analysis**: Uses OpenAI to analyze and structure policy information
- **Comprehensive Output**: Provides eligibility criteria, benefits, action plans, and recommendations
- **RESTful API**: Easy-to-use FastAPI endpoints
- **Relevance Scoring**: Intelligent filtering of policies based on farmer profile

## Prerequisites

- Python 3.8+
- OpenAI API Key
- Tavily API Key (for web search)

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd farmer-policy-agent
```

2. **Install Poetry** (if not already installed)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. **Install dependencies**
```bash
poetry install
```

4. **Set up environment variables**
```bash
cp .env.example .env
```

Edit `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## Project Structure

```
farmer-policy-agent/
├── main.py                 # Main orchestrator
├── app.py                  # FastAPI endpoints
├── agents/
│   ├── config.py          # Configuration management
│   └── curator/
│       └── utils/
│           └── tools/
│               └── search_tool.py  # Web search tool
├── utils/
│   ├── openai_client.py   # OpenAI integration
│   ├── data_processor.py  # Data filtering and processing
│   └── policy_analyzer.py # Output structuring
├── pyproject.toml         # Poetry configuration
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Usage

### Running the API Server

```bash
poetry run python app.py
```

The API will be available at `http://localhost:8000`

### Running the Main Script

```bash
poetry run python main.py
```

### API Endpoints

#### 1. Get Farmer Policies
```http
POST /farmer/policies
```

**Request Body:**
```json
{
  "name": "Rajesh Kumar",
  "location": "Punjab",
  "farm_size_acres": 5.5,
  "crop_types": ["wheat", "rice", "sugarcane"],
  "farming_type": "conventional",
  "annual_income": 300000,
  "land_ownership": "owned"
}
```

**Response:**
```json
{
  "success": true,
  "farmer_name": "Rajesh Kumar",
  "location": "Punjab",
  "relevant_schemes": [...],
  "action_plan": [...],
  "benefits_summary": {...}
}
```

#### 2. Quick Policy Search
```http
POST /farmer/policies/quick
```
Lighter version with top 3 schemes only.

#### 3. Popular Schemes by Location
```http
GET /schemes/popular/{location}
```
Get popular schemes for a specific location.

## Example Usage

### Using the API

```bash
curl -X POST "http://localhost:8000/farmer/policies" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Rajesh Kumar",
       "location": "Punjab",
       "farm_size_acres": 5.5,
       "crop_types": ["wheat", "rice"],
       "farming_type": "conventional",
       "annual_income": 300000,
       "land_ownership": "owned"
     }'
```

### Using Python

```python
from main import FarmerPolicyAgent, FarmerDetails
import asyncio

async def main():
    agent = FarmerPolicyAgent()
    
    farmer = FarmerDetails(
        name="Rajesh Kumar",
        location="Punjab",
        farm_size_acres=5.5,
        crop_types=["wheat", "rice"],
        farming_type="conventional",
        annual_income=300000,
        land_ownership="owned"
    )
    
    result = await agent.get_farmer_policies(farmer)
    print(result)

asyncio.run(main())
```

## Configuration

The application uses environment variables for configuration. Key settings:

- `OPENAI_API_KEY`: Your OpenAI API key
- `TAVILY_API_KEY`: Your Tavily search API key
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-4o-mini)
- `MAX_SEARCH_RESULTS`: Number of search results per query (default: 5)
- `DEBUG`: Enable debug mode (default: False)

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Formatting

```bash
poetry run black .
```

### Type Checking

```bash
poetry run mypy .
```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Output Structure

The agent provides comprehensive output including:

1. **Relevant Schemes**: Top government schemes with eligibility and benefits
2. **Action Plan**: Step-by-step guide to apply for schemes
3. **Benefits Summary**: Estimated financial and non-financial benefits
4. **Recommendations**: Personalized suggestions based on farmer profile
5. **Contact Information**: Relevant government offices and helplines

## Rate Limits and Best Practices

- The application includes rate limiting for API calls
- Search results are cached to improve performance
- Relevance scoring ensures only pertinent policies are analyzed
- Error handling for API failures and invalid inputs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review the example usage above
3. Open an issue on the repository

## Acknowledgments

- OpenAI for GPT models
- Tavily for web search capabilities
- FastAPI for the web framework
- Poetry for dependency management