
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import logging
from typing import Dict, Any, Optional

# Import our AGI system
from src.cognitive_architecture.enhanced_cognitive_architecture import EnhancedCognitiveSystem
from src.utils.logging_config import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger("AGI_Web")

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class WebAGI:
    """Adapter for working with the AGI system through a web interface"""
    def __init__(self):
        self.system = EnhancedCognitiveSystem()
        self.session_history = []
        
    def process_query(self, query: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Process a query from the web interface"""
        try:
            # Process text query
            text_result = self.system.process_input(query, "text")
            
            # If there's a file, process it
            file_result = None
            if file_path:
                file_type = self._determine_file_type(file_path)
                file_result = self.system.process_input(file_path, file_type)
            
            # Save session history
            self.session_history.append({
                'timestamp': datetime.now(),
                'query': query,
                'file_path': file_path,
                'text_result': text_result,
                'file_result': file_result
            })
            
            # Form the response
            response = {
                'query_result': {
                    'processed_text': text_result,
                    'extracted_concepts': [],  # Update this based on your system's output
                    'related_concepts': []  # Update this based on your system's output
                }
            }
            
            # Add file processing results if any
            if file_result:
                response['file_result'] = {
                    'type': file_type,
                    'processed_data': file_result,
                    'related_concepts': []  # Update this based on your system's output
                }
            
            # Add system state
            response['system_state'] = self.system.get_system_state()
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {'error': str(e)}

    def _determine_file_type(self, file_path: str) -> str:
        """Determine file type"""
        ext = file_path.split('.')[-1].lower()
        if ext in {'png', 'jpg', 'jpeg', 'gif'}:
            return 'image'
        elif ext == 'txt':
            return 'text'
        else:
            return 'unknown'

web_agi = WebAGI()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query')
    
    if not query_text:
        return jsonify({'error': 'No query provided'}), 400
    
    result = web_agi.process_query(query_text)
    return jsonify(result)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        query_text = request.form.get('query', '')
        result = web_agi.process_query(query_text, file_path)
        return jsonify(result)
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
