class SystemIntegrator:
    """Компонент для интеграции различных подсистем"""
    def __init__(self, cognitive_system: EnhancedCognitiveArchitecture):
        self.system = cognitive_system
        self.integration_stats = defaultdict(list)
        self.subsystem_performance = {}
        
    def integrate_experience(self, experience_data: Dict[str, Any]):
        """Интегрировать новый опыт во все подсистемы"""
        # Эмоциональная оценка опыта
        emotional_response = self.system.emotional_state.update_emotions({
            'novelty': experience_data.get('novelty', 0.5),
            'learning_success': experience_data.get('confidence', 0.5)
        })
        
        # Создание концепта в графе знаний
        concept = self.system._create_concept(
            experience_data.get('processed_data', {}),
            experience_data.get('modality', 'unknown')
        )
        
        # Обновление долговременной памяти
        memory_id = f"experience_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        importance = self._calculate_experience_importance(experience_data, emotional_response)
        self.system.long_term_memory.store(memory_id, concept, importance)
        
        # Адаптивное обучение
        self.system.adaptive_learning.update_skills(concept, experience_data.get('success_rate', 0.5))
        
        # Генерация творческих идей
        novel_concepts = self.system.creativity.generate_novel_concepts()
        
        return {
            'concept_id': concept.id,
            'memory_id': memory_id,
            'emotional_response': emotional_response,
            'novel_concepts': [nc.id for nc in novel_concepts]
        }
        
    def _calculate_experience_importance(self, experience_data: Dict, emotional_response: Dict) -> float:
        """Расчет важности опыта"""
        factors = {
            'novelty': experience_data.get('novelty', 0.5),
            'emotional_intensity': max(emotional_response.values()),
            'success_rate': experience_data.get('success_rate', 0.5),
            'cognitive_load': experience_data.get('cognitive_load', 0.5)
        }
        
        weights = {
            'novelty': 0.3,
            'emotional_intensity': 0.3,
            'success_rate': 0.2,
            'cognitive_load': 0.2
        }
        
        return sum(value * weights[key] for key, value in factors.items())

class ExperienceSynthesizer:
    """Компонент для синтеза опыта и формирования новых знаний"""
    def __init__(self, cognitive_system: EnhancedCognitiveArchitecture):
        self.system = cognitive_system
        self.synthesis_patterns = {}
        self.learning_sequences = []
        
    def synthesize_experience(self, recent_experiences: List[Dict[str, Any]]):
        """Синтез опыта и формирование новых знаний"""
        # Выявление паттернов в опыте
        patterns = self._identify_patterns(recent_experiences)
        
        # Формирование новых концептов
        new_concepts = self._generate_concepts_from_patterns(patterns)
        
        # Интеграция новых знаний
        integrated_knowledge = self._integrate_new_knowledge(new_concepts)
        
        return integrated_knowledge
        
    def _identify_patterns(self, experiences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Выявление паттернов в опыте"""
        patterns = []
        
        # Группировка по модальностям
        modality_groups = defaultdict(list)
        for exp in experiences:
            modality_groups[exp.get('modality')].append(exp)
            
        # Анализ каждой группы
        for modality, group in modality_groups.items():
            # Поиск повторяющихся элементов
            common_elements = self._find_common_elements(group)
            if common_elements:
                patterns.append({
                    'modality': modality,
                    'elements': common_elements,
                    'frequency': len(group)
                })
                
            # Поиск последовательностей
            sequences = self._find_sequences(group)
            if sequences:
                patterns.append({
                    'modality': modality,
                    'type': 'sequence',
                    'sequences': sequences
                })
                
        return patterns
        
    def _generate_concepts_from_patterns(self, patterns: List[Dict[str, Any]]) -> List[Concept]:
        """Создание новых концептов на основе паттернов"""
        new_concepts = []
        
        for pattern in patterns:
            if pattern.get('type') == 'sequence':
                # Создание концепта последовательности
                concept = Concept(
                    name=f"sequence_{pattern['modality']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    description=f"Sequential pattern in {pattern['modality']} experiences",
                    modality=pattern['modality'],
                    attributes={'sequence_elements': pattern['sequences']}
                )
                new_concepts.append(concept)
            else:
                # Создание концепта паттерна
                concept = Concept(
                    name=f"pattern_{pattern['modality']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    description=f"Common pattern in {pattern['modality']} experiences",
                    modality=pattern['modality'],
                    attributes={'common_elements': pattern['elements']}
                )
                new_concepts.append(concept)
                
        return new_concepts
        
    def _integrate_new_knowledge(self, new_concepts: List[Concept]) -> Dict[str, Any]:
        """Интеграция новых знаний в систему"""
        integration_results = []
        
        for concept in new_concepts:
            # Добавление в граф знаний
            self.system.knowledge_graph.add_concept(concept)
            
            # Поиск связей с существующими концептами
            similar_concepts = self.system.knowledge_graph.find_similar_concepts(concept.embedding)
            
            # Создание связей
            for similar_id, similarity in similar_concepts:
                if similarity > 0.7:  # Порог схожести
                    self.system.knowledge_graph.add_relation(
                        concept.id,
                        similar_id,
                        "derived_from"
                    )
                    
            integration_results.append({
                'concept_id': concept.id,
                'similar_concepts': similar_concepts,
                'integration_time': datetime.now()
            })
            
        return {
            'integrated_concepts': len(new_concepts),
            'integration_details': integration_results
        }
        
    def _find_common_elements(self, experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Поиск общих элементов в опыте"""
        if not experiences:
            return {}
            
        # Получение всех ключей
        all_keys = set().union(*(exp.keys() for exp in experiences))
        
        common_elements = {}
        for key in all_keys:
            # Сбор всех значений для ключа
            values = [exp.get(key) for exp in experiences if key in exp]
            
            # Если значения совпадают более чем в 70% случаев
            if len(values) >= 0.7 * len(experiences):
                most_common = max(set(values), key=values.count)
                if values.count(most_common) >= 0.7 * len(values):
                    common_elements[key] = most_common
                    
        return common_elements
        
    def _find_sequences(self, experiences: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Поиск последовательностей в опыте"""
        sequences = []
        min_sequence_length = 3
        
        # Сортировка по времени
        sorted_exp = sorted(experiences, key=lambda x: x.get('timestamp', datetime.min))
        
        # Поиск последовательностей с помощью скользящего окна
        for i in range(len(sorted_exp) - min_sequence_length + 1):
            window = sorted_exp[i:i + min_sequence_length]
            if self._is_meaningful_sequence(window):
                sequences.append(window)
                
        return sequences
        
    def _is_meaningful_sequence(self, experiences: List[Dict[str, Any]]) -> bool:
        """Проверка последовательности на значимость"""
        if len(experiences) < 2:
            return False
            
        # Проверка временной связности
        for i in range(len(experiences) - 1):
            time1 = experiences[i].get('timestamp', datetime.min)
            time2 = experiences[i + 1].get('timestamp', datetime.min)
            if (time2 - time1).total_seconds() > 3600:  # Разрыв более часа
                return False
                
        # Проверка смысловой связности
        # Например, наличие общих элементов или логической последовательности
        common_elements = self._find_common_elements(experiences)
        if not common_elements:
            return False
            
        return True

# Обновление основного класса
class EnhancedCognitiveArchitecture(CognitiveArchitecture):
    def __init__(self):
        super().__init__()
        self.integrator = SystemIntegrator(self)
        self.synthesizer = ExperienceSynthesizer(self)
        self.recent_experiences = []
        self.max_recent_experiences = 100
        
    def process_input(self, data: Any, modality: str) -> Dict[str, Any]:
        # Базовая обработка
        result = super().process_input(data, modality)
        
        # Добавление временной метки
        experience_data = {
            **result,
            'timestamp': datetime.now(),
            'modality': modality,
            'processed_data': data
        }
        
        # Интеграция опыта
        integration_result = self.integrator.integrate_experience(experience_data)
        
        # Добавление в недавний опыт
        self.recent_experiences.append(experience_data)
        if len(self.recent_experiences) > self.max_recent_experiences:
            self.recent_experiences.pop(0)
            
        # Синтез опыта при накоплении достаточного количества
        if len(self.recent_experiences) >= 10:
            synthesis_result = self.synthesizer.synthesize_experience(self.recent_experiences)
            result['synthesis'] = synthesis_result
            
        result['integration'] = integration_result
        
        return result
