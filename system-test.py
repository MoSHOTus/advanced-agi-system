import logging
from datetime import datetime
import json
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agi_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AGI_Test")

class SystemTester:
    def __init__(self):
        self.system = EnhancedCognitiveArchitecture()
        self.test_results = []
        
    def run_comprehensive_test(self):
        """Запуск полного тестирования системы"""
        logger.info("Starting comprehensive system test")
        
        # Тест 1: Обработка текстовых данных
        self.test_text_processing()
        
        # Тест 2: Обработка изображений
        self.test_image_processing()
        
        # Тест 3: Тест обучения и адаптации
        self.test_learning_adaptation()
        
        # Тест 4: Тест творческих способностей
        self.test_creativity()
        
        # Тест 5: Тест эмоциональной системы
        self.test_emotional_system()
        
        # Тест 6: Тест долговременной памяти
        self.test_long_term_memory()
        
        # Анализ результатов
        self.analyze_results()
        
    def test_text_processing(self):
        """Тест обработки текста"""
        logger.info("Testing text processing")
        
        test_texts = [
            "Artificial intelligence is revolutionizing technology.",
            "Machine learning algorithms can identify patterns.",
            "Neural networks are inspired by biological brains.",
            "Deep learning enables complex pattern recognition."
        ]
        
        for text in test_texts:
            logger.info(f"Processing text: {text}")
            result = self.system.process_input(text, "text")
            self.test_results.append({
                'test_type': 'text_processing',
                'input': text,
                'result': result,
                'timestamp': datetime.now()
            })
            
            # Пауза для наблюдения за развитием системы
            self._analyze_system_state("After text processing")

    def test_image_processing(self):
        """Тест обработки изображений"""
        logger.info("Testing image processing")
        
        # Создаём тестовые изображения (можно заменить на реальные пути к файлам)
        test_images = [
            "test_images/cat.jpg",
            "test_images/dog.jpg",
            "test_images/landscape.jpg"
        ]
        
        for image_path in test_images:
            if os.path.exists(image_path):
                logger.info(f"Processing image: {image_path}")
                result = self.system.process_input(image_path, "image")
                self.test_results.append({
                    'test_type': 'image_processing',
                    'input': image_path,
                    'result': result,
                    'timestamp': datetime.now()
                })
                
                self._analyze_system_state("After image processing")

    def test_learning_adaptation(self):
        """Тест обучения и адаптации"""
        logger.info("Testing learning and adaptation")
        
        # Серия связанных концептов
        concepts = [
            {
                'text': "Cats are mammals",
                'related_to': "animal_classification"
            },
            {
                'text': "Mammals have fur",
                'related_to': "animal_classification"
            },
            {
                'text': "Cats have whiskers",
                'related_to': "cat_features"
            }
        ]
        
        for concept in concepts:
            result = self.system.process_input(concept['text'], "text")
            self.test_results.append({
                'test_type': 'learning_adaptation',
                'input': concept,
                'result': result,
                'timestamp': datetime.now()
            })
            
            # Проверка формирования связей
            self._analyze_system_state("After concept processing")

    def test_creativity(self):
        """Тест творческих способностей"""
        logger.info("Testing creative abilities")
        
        # Генерация новых концептов
        for _ in range(3):
            novel_concepts = self.system.creativity.generate_novel_concepts()
            
            self.test_results.append({
                'test_type': 'creativity',
                'result': {
                    'novel_concepts': [concept.__dict__ for concept in novel_concepts]
                },
                'timestamp': datetime.now()
            })
            
            self._analyze_system_state("After creativity test")

    def test_emotional_system(self):
        """Тест эмоциональной системы"""
        logger.info("Testing emotional system")
        
        # Тестирование различных эмоциональных стимулов
        test_events = [
            {'event': 'new_discovery', 'novelty': 0.9, 'success': 0.8},
            {'event': 'learning_failure', 'novelty': 0.2, 'success': 0.3},
            {'event': 'pattern_recognition', 'novelty': 0.6, 'success': 0.7}
        ]
        
        for event in test_events:
            self.system.emotional_state.update_emotions(event)
            
            self.test_results.append({
                'test_type': 'emotional_system',
                'input': event,
                'result': {
                    'emotions': self.system.emotional_state.emotions.copy(),
                    'mood': self.system.emotional_state.mood
                },
                'timestamp': datetime.now()
            })
            
            self._analyze_system_state("After emotional event")

    def test_long_term_memory(self):
        """Тест долговременной памяти"""
        logger.info("Testing long-term memory")
        
        # Сохранение информации в память
        test_memories = [
            {
                'content': "Important concept about AI",
                'importance': 0.8
            },
            {
                'content': "Basic information",
                'importance': 0.3
            },
            {
                'content': "Critical knowledge",
                'importance': 0.9
            }
        ]
        
        for memory in test_memories:
            memory_id = f"test_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.system.long_term_memory.store(
                memory_id,
                memory['content'],
                memory['importance']
            )
            
            # Проверка извлечения
            retrieved = self.system.long_term_memory.retrieve(memory_id)
            
            self.test_results.append({
                'test_type': 'long_term_memory',
                'input': memory,
                'result': {
                    'stored': memory_id,
                    'retrieved': retrieved
                },
                'timestamp': datetime.now()
            })
            
            self._analyze_system_state("After memory operation")

    def _analyze_system_state(self, checkpoint: str):
        """Анализ состояния системы после каждого этапа"""
        state = self.system.get_system_state()
        logger.info(f"System state at {checkpoint}:")
        logger.info(json.dumps(state, indent=2))
        
        return state

    def analyze_results(self):
        """Анализ результатов тестирования"""
        logger.info("Analyzing test results")
        
        analysis = {
            'total_tests': len(self.test_results),
            'tests_by_type': {},
            'system_growth': self._analyze_system_growth(),
            'emotional_development': self._analyze_emotional_development(),
            'learning_progress': self._analyze_learning_progress()
        }
        
        # Подсчёт тестов по типам
        for result in self.test_results:
            test_type = result['test_type']
            if test_type not in analysis['tests_by_type']:
                analysis['tests_by_type'][test_type] = 0
            analysis['tests_by_type'][test_type] += 1
        
        logger.info("Test analysis complete:")
        logger.info(json.dumps(analysis, indent=2))
        
        return analysis

    def _analyze_system_growth(self):
        """Анализ роста и развития системы"""
        initial_state = self.test_results[0].get('system_state', {})
        final_state = self.test_results[-1].get('system_state', {})
        
        return {
            'knowledge_growth': final_state.get('knowledge_size', 0) - initial_state.get('knowledge_size', 0),
            'experience_growth': final_state.get('experience_level', 0) - initial_state.get('experience_level', 0)
        }

    def _analyze_emotional_development(self):
        """Анализ эмоционального развития"""
        emotional_tests = [r for r in self.test_results if r['test_type'] == 'emotional_system']
        
        if not emotional_tests:
            return {}
            
        return {
            'emotional_range': self._calculate_emotional_range(emotional_tests),
            'mood_stability': self._calculate_mood_stability(emotional_tests)
        }

    def _analyze_learning_progress(self):
        """Анализ прогресса обучения"""
        learning_tests = [r for r in self.test_results if r['test_type'] == 'learning_adaptation']
        
        if not learning_tests:
            return {}
            
        return {
            'concept_formation_rate': len(learning_tests) / max(1, len(self.test_results)),
            'adaptation_score': self._calculate_adaptation_score(learning_tests)
        }

    def _calculate_emotional_range(self, emotional_tests):
        """Расчёт диапазона эмоций"""
        emotions = [test['result']['emotions'] for test in emotional_tests]
        if not emotions:
            return 0
            
        emotion_values = [v for e in emotions for v in e.values()]
        return max(emotion_values) - min(emotion_values)

    def _calculate_mood_stability(self, emotional_tests):
        """Расчёт стабильности настроения"""
        moods = [test['result']['mood'] for test in emotional_tests]
        if len(moods) < 2:
            return 1.0
            
        variations = [abs(moods[i] - moods[i-1]) for i in range(1, len(moods))]
        return 1.0 - sum(variations) / len(variations)

    def _calculate_adaptation_score(self, learning_tests):
        """Расчёт показателя адаптации"""
        if not learning_tests:
            return 0
            
        successive_improvements = 0
        for i in range(1, len(learning_tests)):
            if learning_tests[i]['result'].get('confidence', 0) > learning_tests[i-1]['result'].get('confidence', 0):
                successive_improvements += 1
                
        return successive_improvements / (len(learning_tests) - 1) if len(learning_tests) > 1 else 0

# Запуск тестирования
if __name__ == "__main__":
    tester = SystemTester()
    
    logger.info("Starting system test...")
    tester.run_comprehensive_test()
    logger.info("System test completed!")
    
    # Сохранение результатов
    with open('test_results.json', 'w') as f:
        json.dump(tester.test_results, f, indent=2, default=str)
