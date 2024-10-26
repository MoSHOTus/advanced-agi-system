
from src.testing.system_tester import SystemTester
from src.utils.logging_config import setup_logging
from src.cognitive_architecture.enhanced_cognitive_architecture import EnhancedCognitiveArchitecture, SystemIntegrator

def main():
    setup_logging()
    
    # Initialize the cognitive architecture
    cognitive_system = EnhancedCognitiveArchitecture()
    
    # Initialize the system integrator
    integrator = SystemIntegrator(cognitive_system)
    
    # Run system tests
    tester = SystemTester(cognitive_system, integrator)
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()
