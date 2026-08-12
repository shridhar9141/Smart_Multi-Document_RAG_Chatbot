import importlib
import unittest


class ImportTests(unittest.TestCase):
    def test_services_rag_imports(self):
        module = importlib.import_module("services.rag")
        self.assertTrue(hasattr(module, "RAGService"))


if __name__ == "__main__":
    unittest.main()
