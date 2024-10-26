
from src.testing.system_tester import SystemTester
from src.utils.logging_config import setup_logging

def main():
    setup_logging()
    
    # Initialize the system tester
    tester = SystemTester()
    
    # Run comprehensive system test
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()
