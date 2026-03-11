# Configuration file for the Sphinx documentation builder.
#
# Auto-generated API documentation for Arena Annotator.

import os
import sys

# Add the project root to sys.path so autodoc can import modules
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "Arena Annotator"
copyright = "2026, Bart R.H. Geurten"
author = "Bart R.H. Geurten"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

# Napoleon settings — support Google and NumPy style docstrings
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Mock imports so Sphinx can process modules without their runtime deps
autodoc_mock_imports = ["matplotlib", "PIL", "numpy"]

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "Arena Annotator"
html_static_path = []
