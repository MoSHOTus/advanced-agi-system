class DevelopmentalStage:
    """Представляет стадию развития системы"""
    def __init__(self, name: str, requirements: Dict[str, float]):
        self.name = name
        self.requirements = requirements  # Например: {'experience': 10.0, 'concept_count': 100}
        self.achieved = False
        self.achievement_time = None

    def check_achievement(self, system_state: Dict[str, Any]) -> bool:
        """Проверка достижения стадии развития"""
        if self.achieved:
            return True
            
        requirements_met = all(
            system_state.get(key, 0) >= value 
            for key, value in self.requirements.items()
        )
        
        if requirements_met and not self.achieved:
            self.achieved = True
            self.achievement_time = datetime.now()
            return True
        return False

class CuriosityModule:
    """Модуль любопытства для исследовательского поведения"""
    def __init__(self):
        self.curiosity_level = 1.0
        self.exploration_history = []
        self.interest_areas = defaultdict(float)

    def update_curiosity(self, concept: Concept, similar_concepts: List[Tuple[str, float]]):
        """Обновление уровня любопытства на основе новизны концепта"""
        if not similar_concepts:
            novelty = 1.0
        else:
            # Средняя схожесть с известными концептами
            avg_similarity = sum(sim for _, sim in similar_concepts) / len(similar_concepts)
            novelty = 1.0 - avg_similarity

        # Обновление интереса к области
        self.interest_areas[concept.modality] += novelty * 0.1
        
        # Сохранение истории исследования
        self.exploration_history.append({
            'concept_id': concept.id,
            'novelty': novelty,
            'timestamp': datetime.now()
        })

        # Корректировка общего уровня любопытства
        self.curiosity_level = 0.7 * self.curiosity_level + 0.3 * novelty

    def get_exploration_focus(self) -> str:
        """Определение следующей области для исследования"""
        if not self.interest_areas:
            return None
        return max(self.interest_areas.items(), key=lambda x: x[1])[0]

class RewardSystem:
    """Система вознаграждения для обучения с подкреплением"""
    def __init__(self):
        self.rewards_history = []
        self.value_functions = defaultdict(float)

    def calculate_reward(self, action_result: Dict[str, Any]) -> float:
        """Расчёт награды за действие"""
        reward = 0.0
        
        # Награда за новые концепты
        if action_result.get('new_concept'):
            reward += 1.0
            
        # Награда за установление связей
        reward += len(action_result.get('new_relations', [])) * 0.5
        
        # Награда за точность предсказания
        if 'prediction_accuracy' in action_result:
            reward += action_result['prediction_accuracy'] * 2.0
            
        self.rewards_history.append({
            'reward': reward,
            'action_type': action_result.get('action_type'),
            'timestamp': datetime.now()
        })
        
        return reward

    def update_value_function(self, state: str, reward: float):
        """Обновление функции ценности для состояния"""
        self.value_functions[state] = (
            0.9 * self.value_functions[state] + 0.1 * reward
        )

class MetaCognition:
    """Модуль метапознания для самоанализа и адаптации"""
    def __init__(self):
        self.performance_metrics = defaultdict(list)
        self.learning_strategies = {}
        self.reflection_log = []

    def evaluate_performance(self, action_result: Dict[str, Any]):
        """Оценка производительности системы"""
        metrics = {
            'processing_time': action_result.get('processing_time', 0),
            'memory_usage': action_result.get('memory_size', 0),
            'accuracy': action_result.get('accuracy', 0),
            'novelty': action_result.get('novelty', 0)
        }
        
        for metric, value in metrics.items():
            self.performance_metrics[metric].append(value)

    def reflect(self) -> Dict[str, Any]:
        """Самоанализ системы"""
        reflection = {
            'strengths': self._identify_strengths(),
            'weaknesses': self._identify_weaknesses(),
            'recommendations': self._generate_recommendations()
        }
        
        self.reflection_log.append({
            'reflection': reflection,
            'timestamp': datetime.now()
        })
        
        return reflection

    def _identify_strengths(self) -> List[str]:
        """Определение сильных сторон системы"""
        strengths = []
        
        # Анализ метрик производительности
        for metric, values in self.performance_metrics.items():
            if len(values) > 10:  # Достаточно данных для анализа
                avg_value = sum(values[-10:]) / 10
                if avg_value > 0.7:  # Пороговое значение для "хорошей" производительности
                    strengths.append(f"High {metric}: {avg_value:.2f}")
                    
        return strengths

    def _identify_weaknesses(self) -> List[str]:
        """Определение слабых сторон системы"""
        weaknesses = []
        
        for metric, values in self.performance_metrics.items():
            if len(values) > 10:
                avg_value = sum(values[-10:]) / 10
                if avg_value < 0.3:  # Пороговое значение для "плохой" производительности
                    weaknesses.append(f"Low {metric}: {avg_value:.2f}")
                    
        return weaknesses

    def _generate_recommendations(self) -> List[str]:
        """Генерация рекомендаций по улучшению"""
        recommendations = []
        
        weaknesses = self._identify_weaknesses()
        for weakness in weaknesses:
            if 'accuracy' in weakness:
                recommendations.append("Increase training on current concept type")
            elif 'processing_time' in weakness:
                recommendations.append("Optimize processing pipeline")
            elif 'memory_usage' in weakness:
                recommendations.append("Implement memory cleanup")
                
        return recommendations

# Обновление класса CognitiveArchitecture для использования новых компонентов
class EnhancedCognitiveArchitecture(CognitiveArchitecture):
    def __init__(self):
        super().__init__()
        self.developmental_stages = self._initialize_stages()
        self.curiosity = CuriosityModule()
        self.reward_system = RewardSystem()
        self.metacognition = MetaCognition()

    def _initialize_stages(self) -> Dict[str, DevelopmentalStage]:
        """Инициализация стадий развития"""
        return {
            'infant': DevelopmentalStage('infant', {'experience': 0.0, 'concept_count': 0}),
            'toddler': DevelopmentalStage('toddler', {'experience': 5.0, 'concept_count': 50}),
            'child': DevelopmentalStage('child', {'experience': 20.0, 'concept_count': 200}),
            'adolescent': DevelopmentalStage('adolescent', {'experience': 50.0, 'concept_count': 500}),
            'adult': DevelopmentalStage('adult', {'experience': 100.0, 'concept_count': 1000})
        }

    def process_input(self, data: Any, modality: str) -> Dict[str, Any]:
        """Расширенная обработка входных данных"""
        start_time = datetime.now()
        
        # Базовая обработка
        result = super().process_input(data, modality)
        
        # Обновление любопытства
        self.curiosity.update_curiosity(
            result['concept'], 
            result['similar_concepts']
        )
        
        # Расчет награды
        reward = self.reward_system.calculate_reward({
            'new_concept': True,
            'new_relations': result['similar_concepts'],
            'action_type': 'process_input'
        })
        
        # Метапознание
        processing_time = (datetime.now() - start_time).total_seconds()
        self.metacognition.evaluate_performance({
            'processing_time': processing_time,
            'memory_size': len(self.short_term_memory),
            'novelty': self.curiosity.curiosity_level
        })
        
        # Проверка достижения новых стадий развития
        self._check_developmental_stages()
        
        return {
            **result,
            'processing_time': processing_time,
            'reward': reward,
            'curiosity_level': self.curiosity.curiosity_level,
            'developmental_stage': self._get_current_stage()
        }

    def _check_developmental_stages(self):
        """Проверка и обновление стадий развития"""
        system_state = {
            'experience': self.experience,
            'concept_count': len(self.knowledge_graph.concepts)
        }
        
        for stage in self.developmental_stages.values():
            stage.check_achievement(system_state)

    def _get_current_stage(self) -> str:
        """Получение текущей стадии развития"""
        current_stage = 'infant'
        for name, stage in self.developmental_stages.items():
            if stage.achieved:
                current_stage = name
        return current_stage

    def get_system_state(self) -> Dict[str, Any]:
        """Расширенное состояние системы"""
        base_state = super().get_system_state()
        
        return {
            **base_state,
            'developmental_stage': self._get_current_stage(),
            'curiosity_level': self.curiosity.curiosity_level,
            'interest_areas': dict(self.curiosity.interest_areas),
            'performance_metrics': {
                metric: np.mean(values[-10:]) if values else 0
                for metric, values in self.metacognition.performance_metrics.items()
            },
            'reflection': self.metacognition.reflect()
        }

# Пример использования
if __name__ == "__main__":
    # Инициализация улучшенной системы
    enhanced_system = EnhancedCognitiveArchitecture()
    
    # Обработка нескольких входных данных
    inputs = [
        ("The cat sat on the mat", "text"),
        ("Dogs are loyal pets", "text"),
        ("example_cat.jpg", "image"),
        ("example_dog.jpg", "image")
    ]
    
    for data, modality in inputs:
        print(f"\nProcessing {modality} input: {data}")
        result = enhanced_system.process_input(data, modality)
        print(f"Result summary:")
        print(f"- Processing time: {result['processing_time']:.2f} seconds")
        print(f"- Reward: {result['reward']:.2f}")
        print(f"- Curiosity level: {result['curiosity_level']:.2f}")
        print(f"- Developmental stage: {result['developmental_stage']}")
    
    # Получение итогового состояния системы
    final_state = enhanced_system.get_system_state()
    print("\nFinal system state:")
    print(json.dumps(final_state, indent=2))
