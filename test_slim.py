import unittest
import psutil

class TestSlim(unittest.TestCase):
    def test_memory(self):
        mem = psutil.Process().memory_info().rss / 1024 ** 3
        self.assertLess(mem, 2, f"Memory usage too high: {mem:.2f} GB")

    def test_analyze(self):
        from code_analyzer.main import CodeAnalyzer
        result = CodeAnalyzer().analyze_code('print("test")')
        self.assertTrue(hasattr(result, 'code_quality_score') or hasattr(result, 'report') or hasattr(result, 'documentation'))

if __name__ == '__main__':
    unittest.main() 