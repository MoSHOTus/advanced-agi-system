
from src.testing.system_tester import SystemTester
from src.utils.logging_config import setup_logging
from src.web_interface.app import app as web_app
import threading

def run_tests():
    setup_logging()
    tester = SystemTester()
    tester.run_comprehensive_test()

def run_web_interface():
    web_app.run(debug=True, use_reloader=False)

def main():
    # Run tests
    test_thread = threading.Thread(target=run_tests)
    test_thread.start()

    # Run web interface
    web_thread = threading.Thread(target=run_web_interface)
    web_thread.start()

    # Wait for both threads to complete
    test_thread.join()
    web_thread.join()

if __name__ == "__main__":
    main()
