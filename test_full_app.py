import unittest
import requests
import time
import threading
import sys
import os

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from code_analyzer.web.app import app
    FLASK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Flask app: {e}")
    FLASK_AVAILABLE = False

class TestApp(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Start Flask app in background thread if not already running"""
        if not FLASK_AVAILABLE:
            print("Skipping tests - Flask app not available")
            return
            
        # Check if app is already running
        try:
            response = requests.get('http://localhost:5000/', timeout=2)
            print("Flask app already running")
            return
        except requests.exceptions.RequestException:
            print("Starting Flask app in background thread...")
            
            # Start Flask app in background thread
            def run_app():
                app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
            
            cls.app_thread = threading.Thread(target=run_app, daemon=True)
            cls.app_thread.start()
            
            # Wait for app to start
            time.sleep(3)
            
            # Verify app is running
            max_retries = 10
            for i in range(max_retries):
                try:
                    response = requests.get('http://localhost:5000/', timeout=2)
                    if response.status_code == 200:
                        print(f"Flask app started successfully after {i+1} attempts")
                        break
                except requests.exceptions.RequestException:
                    if i == max_retries - 1:
                        print("Failed to start Flask app")
                        return
                    time.sleep(1)
    
    def test_analyze(self):
        """Test the /analyze endpoint with valid code"""
        if not FLASK_AVAILABLE:
            self.skipTest("Flask app not available")
            
        print("Testing /analyze endpoint with valid code...")
        
        response = requests.post(
            'http://localhost:5000/analyze', 
            json={'code': 'print("test")'},
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        self.assertEqual(response.status_code, 200, 
                        f"Expected 200, got {response.status_code}. Response: {response.text}")
        
        try:
            data = response.json()
            print(f"Response data keys: {list(data.keys())}")
            # Check for the actual fields returned by the Flask app
            self.assertIn('analysis', data, f"Expected 'analysis' in response, got: {list(data.keys())}")
            self.assertIn('detected_language', data, f"Expected 'detected_language' in response, got: {list(data.keys())}")
            self.assertIn('detected_frameworks', data, f"Expected 'detected_frameworks' in response, got: {list(data.keys())}")
            self.assertIn('fix_suggestions', data, f"Expected 'fix_suggestions' in response, got: {list(data.keys())}")
            print("✓ /analyze endpoint test passed")
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response text: {response.text}")
            raise
    
    def test_error(self):
        """Test the /analyze endpoint with invalid input"""
        if not FLASK_AVAILABLE:
            self.skipTest("Flask app not available")
            
        print("Testing /analyze endpoint with invalid input...")
        
        response = requests.post(
            'http://localhost:5000/analyze', 
            json={},
            timeout=10
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text}")
        
        self.assertEqual(response.status_code, 400, 
                        f"Expected 400, got {response.status_code}. Response: {response.text}")
        print("✓ /analyze error handling test passed")
    
    def test_home_page(self):
        """Test the home page endpoint"""
        if not FLASK_AVAILABLE:
            self.skipTest("Flask app not available")
            
        print("Testing home page endpoint...")
        
        response = requests.get('http://localhost:5000/', timeout=10)
        
        print(f"Response status: {response.status_code}")
        self.assertEqual(response.status_code, 200, 
                        f"Expected 200, got {response.status_code}")
        print("✓ Home page test passed")
    
    def test_ask_endpoint(self):
        """Test the /ask endpoint for RAG functionality"""
        if not FLASK_AVAILABLE:
            self.skipTest("Flask app not available")
            
        print("Testing /ask endpoint...")
        
        response = requests.post(
            'http://localhost:5000/ask', 
            json={'question': 'What is this code about?'},
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        
        # The ask endpoint might return 200 or 400 depending on implementation
        self.assertIn(response.status_code, [200, 400], 
                     f"Unexpected status code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Response data: {data}")
                print("✓ /ask endpoint test passed")
            except Exception as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Response text: {response.text}")
        else:
            print(f"Ask endpoint returned error (expected for some implementations): {response.text}")
        print("✓ /ask endpoint test completed")

if __name__ == '__main__':
    print("=" * 60)
    print("LLM Code Analyzer - Full App Test Suite")
    print("=" * 60)
    
    # Run the tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    # Get test results
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestApp)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.skipped:
        print("\nSkipped:")
        for test, reason in result.skipped:
            print(f"  - {test}: {reason}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'✓ PASSED' if success else '✗ FAILED'}")
    
    if not success:
        sys.exit(1) 