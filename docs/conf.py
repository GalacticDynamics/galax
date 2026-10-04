"""Sphinx configuration."""

import importlib.metadata

project = "galax"
copyright = "2023, galax maintainers"
author = "galax maintainers"
version = release = importlib.metadata.version("galax")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
    "**/make_logo.py",  # makes _static/favicon.svg; a tool, not part of the site
]

html_theme = "furo"
html_static_path = ["_static"]
html_favicon = "_static/favicon.svg"  # made by _static/make_logo.py
html_logo = "_static/favicon.svg"

myst_enable_extensions = [
    "colon_fence",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True
