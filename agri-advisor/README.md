# Capital One AI Agriculture Platform

An AI-powered agricultural platform that provides farmers with intelligent assistance through various services including policy recommendations, pest detection, crop price information, and personalized farming guidance.

## 🏗️ Architecture

The platform consists of several microservices:

- **Agents Service** (`agents/`) - Main AI curator service with LangGraph-based conversational AI
- **Data Service** (`data/`) - Data management and storage service with MongoDB integration
- **Indexer Service** (`indexer/`) - Document indexing and vector storage using Qdrant
- **Use Cases** (`usecases/`) - Specialized services for specific agricultural tasks
  - **Policy Fetcher** - Government policy and scheme recommendations
  - **Pest Detection** - AI-powered pest identification from images
  - **Crop Prices** - Real-time crop price information

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Poetry (for dependency management)
- MongoDB instance
- Qdrant vector database
- OpenAI API key
- Various API keys (see Environment Variables section)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd capital1
```

2. **Install Poetry** (if not already installed)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. **Set up environment variables**
Each service requires its own `.env` file. Create the following files:

#### `agents/.env`
```bash
# Database Configuration
DB_HOST=your_database_host
DB_ROUTE=your_database_route
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=capital1_db

# API Keys
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id
TAVILY_API_KEY=your_tavily_api_key

# Vector Database (Qdrant)
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# Embedding Models
TEXT_EMBEDDING_MODEL=your_text_embedding_model
IMAGE_EMBEDDING_MODEL=your_image_embedding_model

# API Configuration
API_HOST=0.0.0.0
API_PORT=8001
```

#### `data/.env`
```bash
# Database Configuration
DB_HOST=your_database_host
DB_ROUTE=your_database_route
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=capital1_db

# API Keys
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id
OPENAI_API_KEY=your_openai_api_key

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

#### `indexer/.env`
```bash
# AWS S3 Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
AWS_BUCKET_NAME=your_aws_bucket_name

# Vector Database (Qdrant)
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# Embedding Models
TEXT_EMBEDDING_MODEL=your_text_embedding_model
IMAGE_EMBEDDING_MODEL=your_image_embedding_model
```

#### `usecases/policy_fetcher/.env`
```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id

# Database Configuration
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=capital1_db

# Vector Database (Qdrant)
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# Embedding Models
TEXT_EMBEDDING_MODEL=your_text_embedding_model
IMAGE_EMBEDDING_MODEL=your_image_embedding_model
```

## 🏃‍♂️ Running the Services

### 1. Start the Data Service

The data service provides data management capabilities and runs on port 8000.

```bash
cd data/
poetry install
poetry run python main.py
```

The service will be available at `http://localhost:8000`

**API Endpoints:**
- `GET /api/data` - Retrieve data from MongoDB collections

### 2. Start the Agents Service

The agents service is the main AI curator that orchestrates various tools and provides conversational AI capabilities. It runs on port 8001.

```bash
cd agents/
poetry install
poetry run python main.py
```

The service will be available at `http://localhost:8001`

**Key Features:**
- Conversational AI with LangGraph
- Tool orchestration (web search, retrieval, weather, prices)
- Task management and user interaction logging
- Multi-modal support (text and images)

### 3. Optional: Run the Indexer Service

The indexer service processes PDF documents and creates vector embeddings for retrieval.

```bash
cd indexer/
poetry install
poetry run python indexer.py
```

**What it does:**
- Processes PDF documents from the `utils/documents/` directory
- Generates metadata and embeddings
- Stores vectors in Qdrant for semantic search

### 4. Optional: Run Use Case Services

#### Policy Fetcher Service
```bash
cd usecases/policy_fetcher/
poetry install
poetry run python app.py
```

#### Pest Detection Service
```bash
cd usecases/pest-detection/
poetry install
poetry run python app.py
```

#### Crop Prices Service
```bash
cd usecases/crop_prices/
poetry install
poetry run python fetcher.py
```

## 📊 Service Details

### Agents Service (Port 8001)
- **Main Components:**
  - `CuratorNode` - Main AI agent orchestrator
  - `TaskManager` - Handles task assignment and tracking
  - `QueryRouter` - Routes queries to appropriate tools
  - **Tools Available:**
    - Web Search Tool (Tavily integration)
    - Retrieval Tool (Qdrant vector search)
    - Price Fetcher Tool
    - Weather Analysis Tool
    - Pest Detection Tool
    - User Data Logger

### Data Service (Port 8000)
- **Main Components:**
  - MongoDB integration for data persistence
  - RESTful API for data operations
  - Support for multiple collections (messages, user_inputs, tasks)
  - CORS enabled for cross-origin requests

### Indexer Service
- **Main Components:**
  - PDF parsing and text extraction
  - Metadata generation using AI
  - Vector embedding generation
  - Qdrant vector database integration
  - Chunking and overlap handling

## 🔧 Development

### Project Structure
```
capital1/
├── agents/           # Main AI service
│   ├── curator/     # Core AI logic
│   ├── config.py    # Configuration
│   ├── main.py      # Service entry point
│   └── routes.py    # API routes
├── data/            # Data management service
│   ├── core/        # Core configurations
│   ├── main.py      # Service entry point
│   └── routes.py    # API routes
├── indexer/         # Document indexing service
│   ├── utils/       # Utility modules
│   └── indexer.py   # Main indexer logic
└── usecases/        # Specialized services
    ├── policy_fetcher/
    ├── pest-detection/
    └── crop_prices/
```

### Adding New Tools

To add a new tool to the agents service:

1. Create a new tool class in `agents/curator/utils/tools/`
2. Implement the required methods following the existing tool patterns
3. Add the tool to the tools list in `agents/routes.py`

### Configuration Management

Each service has its own `config.py` file that loads environment variables. Make sure to update the appropriate config file when adding new environment variables.

## 🐛 Troubleshooting

### Common Issues

1. **Port conflicts**: Make sure ports 8000 and 8001 are available
2. **Missing API keys**: Verify all required API keys are set in your `.env` file
3. **Database connection**: Ensure MongoDB and Qdrant services are running
4. **Dependencies**: Run `poetry install` in each service directory

### Logs

Logs are stored in the `logs/` directory. Check service-specific logs for debugging information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions, please create an issue in the repository or contact the development team.

---

**Note**: This platform is designed to assist farmers with AI-powered agricultural guidance. Ensure all API keys and configurations are properly set up before running the services.
