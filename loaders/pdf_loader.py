import importlib.util
import os

_here = os.path.dirname(__file__)
_impl_path = os.path.abspath(os.path.join(_here, '..', 'Data', 'loader', 'pdf_loader.py'))

spec = importlib.util.spec_from_file_location('loaders._pdf_loader_impl', _impl_path)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

# Re-export PDFLoader and any helpers expected by callers
PDFLoader = getattr(_mod, 'PDFLoader', None)
load_pdf = getattr(_mod, 'load_pdf', None)
