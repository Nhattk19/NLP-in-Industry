"""Page renderers, imported on demand to keep search independent of RAG."""
from importlib import import_module

_EXPORTS = {
    "render_paper_detail_page": "pages.detail",
    "render_home_page": "pages.home",
    "build_results_dataframe": "pages.results",
    "render_result_card": "pages.results",
    "render_results_page": "pages.results",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
