# base.py - Базовые компоненты системы

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
            embedding = concept.embedding.reshape(1, -1)
            faiss.normalize_L2(embedding)
            self.index.add(embedding)

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
# cognitive.py - Когнитивные компоненты системы

from base import *

class EmotionalState:
    """Эмоциональная система"""
    def __init__(self):
        self.emotions = {
            'curiosity': 0.5,
            'satisfaction': 0.5,
            'frustration': 0.0,
            'excitement': 0.5
        }
        self.mood = 0.5
        self.emotional_history = []

    def update_emotions(self, event: Dict[str, Any]):
        if 'learning_success' in event:
            success_rate = event['learning_success']
            self.emotions['satisfaction'] = 0.8 * self.emotions['satisfaction'] + 0.2 * success_rate
            self.emotions['frustration'] = 0.8 * self.emotions['frustration'] + 0.2 * (1 - success_rate)

        if 'novelty' in event:
            novelty = event['novelty']
            self.emotions['curiosity'] = 0.8 * self.emotions['curiosity'] + 0.2 * novelty
            self.emotions['excitement'] = 0.8 * self.emotions['excitement'] + 0.2 * novelty

        self.mood = sum(self.emotions.values()) / len(self.emotions)
        self.emotional_history.append({
            'timestamp': datetime.now(),
            'emotions': self.emotions.copy(),
            'mood': self.mood
        })

    def get_dominant_emotion(self) -> Tuple[str, float]:
        return max(self.emotions.items(), key=lambda x: x[1])

class AdaptiveLearningModule:
    """Модуль адаптивного обучения"""
    def __init__(self):
        self.skills = defaultdict(float)
        self.learning_strategies = {}
        self.skill_hierarchy = nx.DiGraph()
        self.development_goals = []
        
    def update_skills(self, concept: Concept, success_rate: float):
        # Обновление основного навыка
        primary_skill = concept.modality
        self.skills[primary_skill] = 0.9 * self.skills[primary_skill] + 0.1 * success_rate
        
        # Обновление связанных навыков
        related_skills = self._identify_related_skills(concept)
        for skill, relation_strength in related_skills.items():
            self.skills[skill] += 0.05 * success_rate * relation_strength
        
        self._update_skill_hierarchy(primary_skill, related_skills)

    def _identify_related_skills(self, concept: Concept) -> Dict[str, float]:
        related_skills = {}
        
        if concept.modality == "text":
            related_skills["language_processing"] = 1.0
            related_skills["pattern_recognition"] = 0.5
        elif concept.modality == "image":
            related_skills["visual_processing"] = 1.0
            related_skills["pattern_recognition"] = 0.7
            
        return related_skills

    def suggest_learning_path(self) -> List[str]:
        if not self.skills:
            return ["basic_perception", "pattern_recognition"]
            
        weak_skills = sorted(
            self.skills.items(),
            key=lambda x: x[1]
        )[:3]
        
        learning_path = []
        for skill, _ in weak_skills:
            prerequisites = list(nx.ancestors(self.skill_hierarchy, skill))
            learning_path.extend([p for p in prerequisites if p not in learning_path])
            learning_path.append(skill)
            
        return learning_path

class CreativityModule:
    """Модуль креативности"""
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.novel_combinations = []
        self.creativity_patterns = {}
        
    def generate_novel_concepts(self) -> List[Concept]:
        base_concepts = self._select_diverse_concepts(3)
        if not base_concepts:
            return []
            
        novel_concepts = []
        
        for strategy in ['merge', 'analogize', 'transform']:
            concept = self._apply_creative_strategy(strategy, base_concepts)
            if concept:
                novel_concepts.append(concept)
                
        return novel_concepts

    def _select_diverse_concepts(self, n: int) -> List[Concept]:
        if not self.knowledge_graph.concepts:
            return []
            
        concepts = list(self.knowledge_graph.concepts.values())
        selected = [concepts[0]]
        
        while len(selected) < n and len(selected) < len(concepts):
            max_distance = 0
            next_concept = None
            
            for concept in concepts:
                if concept in selected:
                    continue
                    
                min_distance = float('inf')
                for selected_concept in selected:
                    if concept.embedding is not None and selected_concept.embedding is not None:
                        distance = np.linalg.norm(concept.embedding - selected_concept.embedding)
                        min_distance = min(min_distance, distance)
                
                if min_distance > max_distance:
                    max_distance = min_distance
                    next_concept = concept
                    
            if next_concept:
                selected.append(next_concept)
            else:
                break
                
        return selected

    def _apply_creative_strategy(self, strategy: str, concepts: List[Concept]) -> Optional[Concept]:
        if strategy == 'merge':
            embeddings = [c.embedding for c in concepts if c.embedding is not None]
            if embeddings:
                new_embedding = np.mean(embeddings, axis=0)
                return Concept(
                    name=f"merged_{'_'.join(c.name for c in concepts)}",
                    embedding=new_embedding,
                    modality='hybrid'
                )
                
        elif strategy == 'analogize':
            if len(concepts) >= 2:
                c1, c2 = concepts[:2]
                relation = self._find_relation(c1, c2)
                if relation:
                    return Concept(
                        name=f"analogy_{c1.name}_{c2.name}",
                        description=f"Analogy based on {relation}",
                        modality='abstract'
                    )
                    
        elif strategy == 'transform':
            if concepts:
                base = concepts[0]
                if base.embedding is not None:
                    transform = np.random.normal(0, 0.1, base.embedding.shape)
                    new_embedding = base.embedding + transform
                    return Concept(
                        name=f"transformed_{base.name}",
                        embedding=new_embedding,
                        modality=base.modality
                    )
                    
        return None

    def _find_relation(self, c1: Concept, c2: Concept) -> Optional[str]:
        for relation_type, related_ids in c1.relations.items():
            if c2.id in related_ids:
                return relation_type
        return None
