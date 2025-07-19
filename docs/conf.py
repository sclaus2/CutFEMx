# Configuration file for the Sphinx documentation builder.
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Get version from the Python package
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

try:
    import cutfemx

    release = cutfemx.__version__
    version = ".".join(cutfemx.__version__.split(".")[:2])  # X.Y format
except ImportError:
    # Fallback if package not built
    release = "0.1.0"
    version = "0.1"

# -- Project information -----------------------------------------------------
project = "CutFEMx"
copyright = "2025, ONERA"
author = "Susanne Claus"

# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# extensions.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
    "numpydoc",
]

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Add any paths that contain templates here, relative to this directory.
# Using theme defaults since we don't have custom templates
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "__pycache__"]

# The suffix(es) of source filenames.
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.
html_theme = "pydata_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.
html_theme_options = {
    "logo": {
        "image_light": "_static/cutfemx_logo_light.png",
        "image_dark": "_static/cutfemx_logo_dark.png",
        "text": "CutFEMx",
        "alt_text": "CutFEMx - Cut Finite Element Methods for FEniCSx",
    },
    "navbar_start": [],
    "navbar_center": ["navbar-nav"],  # Use automatic navigation from toctrees
    "navbar_end": ["navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "show_navbar_depth": 2,  # Show 2 levels to enable dropdowns
    "header_links_before_dropdown": 3,  # Show 3 main items before dropdown
    "primary_sidebar_end": [],
    "show_nav_level": 2,  # Show 2 levels in sidebar
    "navigation_with_keys": True,
    "collapse_navigation": False,
    "navigation_depth": 2,
    "sidebar_includehidden": True,  # Include hidden toctrees in navigation
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "footer_end": [],
    "show_prev_next": False,
    "search_bar_text": "Search the docs...",
    "show_toc_level": 2,
    "announcement": "",
    "use_edit_page_button": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/sclaus2/CutFEMx",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
}

# Edit page configuration
html_context = {
    "github_user": "sclaus2",
    "github_repo": "CutFEMx",
    "github_version": "main",
    "doc_path": "docs",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom CSS files
html_css_files = [
    "custom_pydata.css",
]

# Custom JS files
html_js_files = [
    "custom_pydata.js",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
pygments_dark_style = "monokai"

# HTML title
html_title = f"{project} {version}"

# HTML favicon
html_favicon = "_static/cutfemx_logo.png"

# -- Options for HTMLHelp output ---------------------------------------------
# Output file base name for HTML help builder.
htmlhelp_basename = "CutFEMxdoc"

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "fncychap": "\\usepackage[Bjornstrup]{fncychap}",
    "printindex": "\\footnotesize\\raggedright\\printindex",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass).
latex_documents = [
    (
        master_doc,
        "CutFEMx.tex",
        "CutFEMx Documentation",
        "CutFEMx Developers",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------
# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "cutfemx", "CutFEMx Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------
# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "CutFEMx",
        "CutFEMx Documentation",
        author,
        "CutFEMx",
        "Cut Finite Element Methods for FEniCSx",
        "Miscellaneous",
    ),
]

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------
# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "dolfinx": ("https://docs.fenicsproject.org/dolfinx/v0.9.0/python/", None),
    "ufl": ("https://docs.fenicsproject.org/ufl/main/", None),
    "basix": ("https://docs.fenicsproject.org/basix/v0.9.0/python/", None),
}

# -- Options for autodoc extension -------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

# Mock imports for modules that might not be available during doc build
autodoc_mock_imports = [
    "dolfinx",
    "ufl",
    "basix",
    "petsc4py",
    "mpi4py",
    "numpy",
    "scipy",
    "matplotlib",
    "pyvista",
    "cutfemx_cpp",
    "cutfemx.cutfemx_cpp",
]

# Autosummary settings
autosummary_generate = False  # Disable autosummary generation
autosummary_imported_members = False

# -- Options for napoleon extension ------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# -- Options for sphinx-gallery extension -----------------------------------
sphinx_gallery_conf = {
    "examples_dirs": [],  # Disable for now to avoid import issues
    "gallery_dirs": [],  # Disable for now
    "plot_gallery": False,  # Disable gallery generation
}

# -- Options for copybutton extension ----------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True

# -- Options for numpydoc extension ------------------------------------------
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = True
numpydoc_class_members_toctree = False
