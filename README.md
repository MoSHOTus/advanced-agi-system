
# Advanced AGI System

This project implements an advanced AGI (Artificial General Intelligence) system with a web interface for interaction and visualization.

## Project Structure

```
.
├── src
│   ├── cognitive_architecture
│   │   ├── agi_core.py
│   │   └── enhanced_cognitive_architecture.py
│   ├── testing
│   │   └── system_tester.py
│   ├── utils
│   │   ├── logging_config.py
│   │   └── visualization.py
│   └── web_interface
│       ├── app.py
│       └── templates
│           └── index.html
├── data
│   └── test_images
├── tests
├── main.py
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On Unix or MacOS: `source venv/bin/activate`
4. Install the required packages: `pip install -r requirements.txt`

## Running the System

To run the system, including both the comprehensive tests and the web interface, simply execute:

```
python main.py
```

This will start the system tests in one thread and the web interface in another. You can access the web interface by opening a web browser and navigating to `http://localhost:5000`.

## Components

- **Cognitive Architecture**: The core AGI system implementation.
- **System Tester**: Comprehensive tests for the AGI system.
- **Web Interface**: A Flask-based web application for interacting with the AGI system and visualizing its state and outputs.

## Web Interface

The web interface provides the following features:
- Query input for text-based interactions with the AGI system
- File upload for processing images or other supported file types
- Visualization of system stats, knowledge graphs, and emotional indicators
- Display of processed results and extracted concepts

To use the web interface, enter your query in the text area, optionally upload a file, and click "Process Query". The results will be displayed in various sections of the interface.

