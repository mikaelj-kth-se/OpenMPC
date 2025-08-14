import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Ensures Sphinx finds openmpc
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "examples")))

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
    'sphinx.ext.autodoc',     # Auto-generates API docs
    'sphinx.ext.autosummary', # Generates summaries of modules/classes
    'sphinx.ext.napoleon',    # Supports Google-style docstrings
    'sphinx.ext.viewcode',    # Adds links to source code
    'sphinx.ext.mathjax',     # Render LaTeX equations in HTML
    'myst_nb'                 # Enables Jupyter notebooks rendering (no Pandoc needed)
]

# MyST-NB settings
nb_execution_mode = 'off'  # Don't execute notebooks during build
nb_render_markdown_format = 'myst'  # MyST parser for markdown cells

# Optional: Control warnings if cells fail
nb_execution_allow_errors = True

myst_heading_anchors = 2  # still allows anchor links from H2+
myst_enable_extensions = ["colon_fence"]
myst_heading_h1_level = 2  # treat notebook H1s as H2


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
html_logo = '_static/logo.png'
html_theme_options = {
    'logo_only': True,
    'display_version': True,
}
