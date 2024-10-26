
from .agi_core import EnhancedCognitiveArchitecture, TextProcessor, ImageProcessor, SemanticNetwork, ConceptualGraph, EpisodicMemory, ProceduralMemory

class EnhancedCognitiveSystem(EnhancedCognitiveArchitecture):
    def __init__(self):
        super().__init__()
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()

    def process_input(self, input_data, input_type):
        if input_type == "text":
            return self.text_processor.process(input_data)
        elif input_type == "image":
            return self.image_processor.process(input_data)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

    def learn_and_adapt(self, input_data, input_type):
        processed_data = self.process_input(input_data, input_type)
        self.update_semantic_network(processed_data)
        self.update_conceptual_graph(processed_data)
        self.update_episodic_memory(processed_data)
        self.update_procedural_memory(processed_data)
        return processed_data

    def get_system_state(self):
        return {
            "semantic_network_size": len(self.semantic_network.graph.nodes),
            "conceptual_graph_size": len(self.conceptual_graph.concepts),
            "episodic_memory_size": len(self.episodic_memory.episodes),
            "procedural_memory_size": len(self.procedural_memory.procedures)
        }
