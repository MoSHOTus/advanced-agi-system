import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import spacy
import faiss
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import logging
from collections import defaultdict, OrderedDict
import json
from uuid import uuid4
from sklearn.cluster import DBSCAN
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AdvancedAGI")

class ModalityProcessor:
    """Базовый класс для обработки различных модальностей данных"""
    def __init__(self):
        self.name = self.__class__.__name__
        self.processing_history = []

    def process(self, data: Any) -> Dict[str, Any]:
        raise NotImplementedError

class TextProcessor(ModalityProcessor):
    """Обработчик текстовых данных"""
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.nlp = spacy.load("en_core_web_sm")

    def process(self, text: str) -> Dict[str, Any]:
        # BERT embedding
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        bert_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        # SpaCy analysis
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        return {
            'text': text,
            'embedding': bert_embedding,
            'entities': entities,
            'tokens': [token.text for token in doc],
            'pos_tags': [(token.text, token.pos_) for token in doc]
        }

class ImageProcessor(ModalityProcessor):
    """Обработчик изображений"""
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def process(self, image_path: str) -> Dict[str, Any]:
        image = Image.open(image_path)
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            features = self.model(input_batch)
            
        return {
            'image_path': image_path,
            'features': features.squeeze().numpy(),
            'size': image.size,
            'mode': image.mode
        }

@dataclass
class Concept:
    """Расширенное представление концепта"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    modality: str = "unknown"
    embedding: Optional[np.ndarray] = None
    relations: Dict[str, List[str]] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    creation_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    
    def update(self, **kwargs):
        """Обновление атрибутов концепта"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = datetime.now()
        self.usage_count += 1

class KnowledgeGraph:
    """Расширенный граф знаний с поддержкой различных модальностей"""
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.graph = nx.MultiDiGraph()
        self.index = None
        self.dimension = 768  # Размерность BERT-эмбеддингов
        self._initialize_index()
        
    def _initialize_index(self):
        """Инициализация FAISS индекса"""
        self.index = faiss.IndexFlatL2(self.dimension)
        
    def add_concept(self, concept: Concept):
        """Добавление концепта в граф"""
        self.concepts[concept.id] = concept
        self.graph.add_node(concept.id, **concept.__dict__)
        
        if concept.embedding is not None:
            if self.index.ntotal == 0:
                self.index.add(concept.embedding.reshape(1, -1))
            else:
                faiss.normalize_L2(concept.embedding.reshape(1, -1))
                self.index.add(concept.embedding.reshape(1, -1))

    def add_relation(self, source_id: str, target_id: str, relation_type: str):
        """Добавление связи между концептами"""
        if source_id in self.concepts and target_id in self.concepts:
            self.graph.add_edge(source_id, target_id, type=relation_type)
            self.concepts[source_id].relations.setdefault(relation_type, []).append(target_id)

    def find_similar_concepts(self, embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Поиск похожих концептов"""
        if self.index.ntotal == 0:
            return []
            
        embedding = embedding.reshape(1, -1)
        faiss.normalize_L2(embedding)
        D, I = self.index.search(embedding, k)
        
        results = []
        for idx, distance in zip(I[0], D[0]):
            concept_id = list(self.concepts.keys())[idx]
            results.append((concept_id, float(distance)))
        return results

    def get_concept_neighborhood(self, concept_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Получение окружения концепта"""
        if concept_id not in self.concepts:
            return {}
            
        neighborhood = nx.ego_graph(self.graph, concept_id, radius=max_depth)
        return {
            'nodes': list(neighborhood.nodes()),
            'edges': list(neighborhood.edges(data=True)),
            'central_concept': self.concepts[concept_id].__dict__
        }

class CognitiveArchitecture:
    """Архитектура когнитивной системы"""
    def __init__(self):
        self.knowledge_graph = KnowledgeGraph()
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.short_term_memory: List[Dict] = []
        self.attention_weights: Dict[str, float] = defaultdict(float)
        self.learning_rate = 0.1
        self.experience = 0.0

    def process_input(self, data: Any, modality: str) -> Dict[str, Any]:
        """Обработка входных данных различных модальностей"""
        if modality == "text":
            processed = self.text_processor.process(data)
        elif modality == "image":
            processed = self.image_processor.process(data)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        # Создание концепта из обработанных данных
        concept = self._create_concept(processed, modality)
        
        # Поиск похожих концептов
        similar_concepts = self.knowledge_graph.find_similar_concepts(concept.embedding)
        
        # Обновление графа знаний
        self._update_knowledge(concept, similar_concepts)
        
        # Обновление краткосрочной памяти
        self._update_memory(concept)
        
        return {
            'concept': concept.__dict__,
            'similar_concepts': similar_concepts,
            'attention': dict(self.attention_weights)
        }

    def _create_concept(self, processed_data: Dict[str, Any], modality: str) -> Concept:
        """Создание концепта из обработанных данных"""
        concept = Concept(
            name=f"{modality}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            modality=modality,
            embedding=processed_data.get('embedding') or processed_data.get('features'),
            attributes=processed_data
        )
        return concept

    def _update_knowledge(self, concept: Concept, similar_concepts: List[Tuple[str, float]]):
        """Обновление графа знаний"""
        self.knowledge_graph.add_concept(concept)
        
        # Создание связей с похожими концептами
        for similar_id, similarity in similar_concepts:
            if similarity > 0.8:  # Порог схожести
                self.knowledge_graph.add_relation(concept.id, similar_id, "similar_to")
                
        # Обновление опыта системы
        self.experience += self.learning_rate
        
        # Корректировка скорости обучения
        self.learning_rate *= 0.999  # Постепенное уменьшение скорости обучения

    def _update_memory(self, concept: Concept):
        """Обновление краткосрочной памяти"""
        self.short_term_memory.append({
            'concept_id': concept.id,
            'timestamp': datetime.now(),
            'attention': self.calculate_attention(concept)
        })
        
        # Ограничение размера краткосрочной памяти
        if len(self.short_term_memory) > 100:
            self.short_term_memory.pop(0)

    def calculate_attention(self, concept: Concept) -> float:
        """Расчет внимания к концепту"""
        # Базовое внимание на основе новизны
        attention = 1.0
        
        # Корректировка на основе похожих концептов
        similar = self.knowledge_graph.find_similar_concepts(concept.embedding, k=1)
        if similar:
            _, similarity = similar[0]
            attention *= (1 - similarity)  # Меньше внимания к похожим концептам
            
        # Сохранение значения внимания
        self.attention_weights[concept.id] = attention
        return attention

    def get_system_state(self) -> Dict[str, Any]:
        """Получение текущего состояния системы"""
        return {
            'experience_level': self.experience,
            'learning_rate': self.learning_rate,
            'memory_size': len(self.short_term_memory),
            'knowledge_size': len(self.knowledge_graph.concepts),
            'attention_focus': dict(sorted(
                self.attention_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])  # Топ-5 концептов по вниманию
        }

# Пример использования
if __name__ == "__main__":
    # Инициализация системы
    cognitive_system = CognitiveArchitecture()
    
    # Пример обработки текста
    text_input = "Artificial intelligence is transforming the world through machine learning and neural networks."
    text_result = cognitive_system.process_input(text_input, "text")
    print("Processed text:", json.dumps(text_result, indent=2))
    
    # Пример обработки изображения
    image_result = cognitive_system.process_input("example_image.jpg", "image")
    print("Processed image:", json.dumps(image_result, indent=2))
    
    # Получение состояния системы
    system_state = cognitive_system.get_system_state()
    print("System state:", json.dumps(system_state, indent=2))
