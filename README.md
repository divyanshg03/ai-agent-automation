# AI Agents for Medical Diagnostics
## The code was tested on google colab enabled with T4GPU accelerator,  GPUs with VRAM of <16GB will not be able to run the code 
## Overview
This repository contains a framework for medical diagnostics using AI agents specialized in different medical domains. The system processes patient medical reports through multiple specialized agents (Cardiologist, Psychologist, Pulmonologist) and synthesizes their findings into comprehensive diagnostic recommendations.

## Features
- **Multi-agent diagnostic system** with domain-specific AI specialists
- **Concurrent processing** of reports for efficient analysis
- **Web interface** for easy interaction with the system
- **API endpoints** for integration with other healthcare systems
- **Evaluation framework** for measuring agent performance

## Architecture
The system uses a team of specialized AI agents, each trained for a specific medical domain:
- **Cardiologist**: Analyzes cardiac symptoms and test results
- **Psychologist**: Evaluates psychological aspects of patient cases
- **Pulmonologist**: Assesses respiratory conditions and related symptoms
- **Multidisciplinary Team**: Synthesizes individual specialist insights into comprehensive recommendations

## Setup and Installation

### Prerequisites
- Python 3.8+
- Flask
- LangChain
- openai/gpt-oss-120b

### Installation
1. Clone the repository
   ```bash
   git clone https://github.com/divyanshg03/ai-agent-automation.git
   cd ai-agent-automation
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables
   ```
   # Create a .env file with your API key
   GROQ_API_KEY=your_api_key_here
   ```

## Usage

### Running the Web Interface
```bash
python flask_app.py
```
Then open http://localhost:5000 in your browser.

### API Usage
```python
import requests
import json

url = "http://localhost:5000/api/analyze"
payload = {"medical_report": "Patient reports chest pain and palpitations..."}
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(payload), headers=headers)
results = response.json()
print(results["results"]["final_diagnosis"])
```

### Command Line Interface
```bash
python Main.py
```
Follow the prompts to enter a medical report for analysis.

## Evaluation
The system includes an evaluation framework (`evaluate_agent.py`) that measures:
- Semantic similarity between input and agent output
- Medical entity coverage
- Readability of agent responses
- Overall composite score

## Project Structure
```
├── Main.py                     # Command-line interface
├── flask_app.py                # Web server and API endpoints
├── Utils/
│   ├── Agents.py               # AI agent implementations
│   └── evaluate_agent.py       # Evaluation metrics
├── Medical Reports/            # Sample medical reports
├── templates/                  # Web interface templates
└── requirements.txt            # Project dependencies
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- LangChain for the agent framework
- Google Generative AI for the Gemini model
- Flask for the web interface
