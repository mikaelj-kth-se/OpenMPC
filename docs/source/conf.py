import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Ensures Sphinx finds openmpc

# Make sure Sphinx finds examples
examples_dir = os.path.abspath('../../examples')
if examples_dir not in sys.path:
    sys.path.insert(0, examples_dir)
    

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'openmpc'
copyright = '2025, Mikael Johansson, Pedro Roque and Gregorio Marchesini'
author = 'Mikael Johansson, Pedro Roque and Gregorio Marchesini'
release = '0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',    # Auto-generates API docs
    "sphinx.ext.autosummary", # Generates summaries of modules/classes
    'sphinx.ext.napoleon',   # Supports Google-style docstrings
    'sphinx.ext.viewcode',   # Adds links to source code
    'sphinx.ext.mathjax',
    'myst_parser',           # Enables Markdown (.md) support
    'nbsphinx'               # Enables Jupyter notebooks rendering
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": False,
    "module-first": True,  # Show the module name at the top
}


# Avoid showing module names in front of function definitions
add_module_names = False 


nbsphinx_execute = 'never'  # Avoids executing notebooks during doc build


templates_path = ['_templates']
exclude_patterns = ['../build', '../testing', '.../openmpc.egg-info', '../OpenMPC.egg-info','_build']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
# logo
html_logo = '_static/mpc_logo.svg'
html_theme_options = {
    'logo_only': True,
    'display_version': True,
}
