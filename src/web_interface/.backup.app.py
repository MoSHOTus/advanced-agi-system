from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import logging
from typing import Dict, Any, Optional

# Импортируем нашу AGI систему
from base import *
from cognitive import *
from memory import LongTermMemory
from integration import SystemIntegrator, ExperienceSynthesizer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agi_web.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AGI_Web")

app = Flask(__name__)

# Конфигурация
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Создаем папку для загрузок, если её нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class WebAGI:
    """Адаптер для работы AGI системы через веб-интерфейс"""
    def __init__(self):
        self.system = EnhancedCognitiveArchitecture()
        self.session_history = []
        
    def process_query(self, query: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Обработка запроса от веб-интерфейса"""
        try:
            # Обработка текстового запроса
            text_result = self.system.process_input(query, "text")
            
            # Если есть файл, обрабатываем его
            file_result = None
            if file_path:
                file_type = self._determine_file_type(file_path)
                file_result = self.system.process_input(file_path, file_type)
            
            # Сохраняем историю сессии
            self.session_history.append({
                'timestamp': datetime.now(),
                'query': query,
                'file_path': file_path,
                'text_result': text_result,
                'file_result': file_result
            })
            
            # Формируем ответ
            response = {
                'query_result': {
                    'processed_text': text_result['concept']['name'],
                    'extracted_concepts': [
                        c['name'] for c in text_result.get('similar_concepts', [])
                    ],
                    'related_concepts': [
                        c for c in self.system.knowledge_graph.get_concept_neighborhood(
                            text_result['concept']['id']
                        ).get('nodes', [])
                    ]
                }
            }
            
            # Добавляем результаты обработки файла, если есть
            if file_result:
                response['file_result'] = {
                    'type': file_type,
                    'processed_data': file_result['concept']['name'],
                    'related_concepts': [
                        c['name'] for c in file_result.get('similar_concepts', [])
                    ]
                }
            
            # Добавляем состояние системы
            response['system_state'] = self.system.get_system_state()
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {'error': str(e)}

    def _determine_file_type(self, file_path: str) -> str:
        """Определение типа файла"""
        ext = file_path.split('.')[-1].lower()
        if ext in {'png', 'jpg', 'jpeg', 'gif'}:
            return 'image'
        elif ext in {'txt', 'pdf'}:
            return 'text'
        else:
            raise ValueError(f"Unsupported file type: {ext}")

# Создаем экземпляр AGI системы
agi_system = WebAGI()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    try:
        query = request.form.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        # Обработка файла, если он есть
        file_path = None
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
        
        # Обработка запроса через AGI систему
        result = agi_system.process_query(query, file_path)
        
        # Если был файл, добавляем его имя в ответ
        if file_path:
            result['filename'] = os.path.basename(file_path)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/system_state')
def get_system_state():
    try:
        state = agi_system.system.get_system_state()
        return jsonify(state)
    except Exception as e:
        logger.error(f"Error getting system state: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting AGI Web Interface...")
    app.run(debug=True, host='0.0.0.0', port=5000)
