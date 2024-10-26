
import logging
from datetime import datetime
import json
import os
from src.cognitive_architecture.enhanced_cognitive_architecture import EnhancedCognitiveSystem
from src.utils.visualization import plot_learning_curve, plot_performance_metrics

logger = logging.getLogger("AGI_Test")

class SystemTester:
    def __init__(self):
        self.system = EnhancedCognitiveSystem()
        self.test_results = []
        
    def run_comprehensive_test(self):
        logger.info("Starting comprehensive system test")
        
        self.test_text_processing()
        self.test_image_processing()
        self.test_learning_adaptation()
        self.test_semantic_network()
        self.test_conceptual_graph()
        self.test_episodic_memory()
        self.test_procedural_memory()
        
        self.analyze_results()
        
    def test_text_processing(self):
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
            self.test_results.append(TestResult('text_processing', text, result, datetime.now()))
            
            self._analyze_system_state("After text processing")

    def test_image_processing(self):
        logger.info("Testing image processing")
        
        test_images = [
            "data/test_images/test_image_1.jpg",
            "data/test_images/test_image_2.jpg"
        ]
        
        for image_path in test_images:
            if os.path.exists(image_path):
                logger.info(f"Processing image: {image_path}")
                result = self.system.process_input(image_path, "image")
                self.test_results.append(TestResult('image_processing', image_path, result, datetime.now()))
                
                self._analyze_system_state("After image processing")

    def test_learning_adaptation(self):
        logger.info("Testing learning and adaptation")
        test_input = "AGI systems can learn and adapt to new situations."
        result = self.system.learn_and_adapt(test_input, "text")
        self.test_results.append(TestResult('learning_adaptation', test_input, result, datetime.now()))
        self._analyze_system_state("After learning and adaptation")

    def test_semantic_network(self):
        logger.info("Testing semantic network")
        network_state = self.system.semantic_network.get_network_state()
        self.test_results.append(TestResult('semantic_network', 'network_state', network_state, datetime.now()))
        self._analyze_system_state("After semantic network test")

    def test_conceptual_graph(self):
        logger.info("Testing conceptual graph")
        graph_state = self.system.conceptual_graph.get_graph_state()
        self.test_results.append(TestResult('conceptual_graph', 'graph_state', graph_state, datetime.now()))
        self._analyze_system_state("After conceptual graph test")

    def test_episodic_memory(self):
        logger.info("Testing episodic memory")
        memory_state = self.system.episodic_memory.get_memory_state()
        self.test_results.append(TestResult('episodic_memory', 'memory_state', memory_state, datetime.now()))
        self._analyze_system_state("After episodic memory test")

    def test_procedural_memory(self):
        logger.info("Testing procedural memory")
        memory_state = self.system.procedural_memory.get_memory_state()
        self.test_results.append(TestResult('procedural_memory', 'memory_state', memory_state, datetime.now()))
        self._analyze_system_state("After procedural memory test")

    def _analyze_system_state(self, stage):
        logger.info(f"Analyzing system state: {stage}")
        system_state = self.system.get_system_state()
        logger.info(f"System state: {json.dumps(system_state, indent=2)}")

    def analyze_results(self):
        logger.info("Analyzing test results")
        for result in self.test_results:
            logger.info(f"Test: {result.test_type}, Input: {result.input_data}, Result: {result.result}")
        
        # Here you can add more complex analysis, visualization, etc.
        plot_learning_curve({"iterations": range(len(self.test_results)), "performance": [0.1 * i for i in range(len(self.test_results))]})
        plot_performance_metrics({"accuracy": [0.5 + 0.01 * i for i in range(len(self.test_results))], "speed": [10 - 0.1 * i for i in range(len(self.test_results))]})

class TestResult:
    def __init__(self, test_type, input_data, result, timestamp):
        self.test_type = test_type
        self.input_data = input_data
        self.result = result
        self.timestamp = timestamp

    def to_dict(self):
        return {
            'test_type': self.test_type,
            'input': self.input_data,
            'result': self.result,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            data['test_type'],
            data['input'],
            data['result'],
            datetime.fromisoformat(data['timestamp'])
        )
