
from src.testing.system_tester import SystemTester
from src.utils.logging_config import setup_logging

def main():
    setup_logging()
    tester = SystemTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()
