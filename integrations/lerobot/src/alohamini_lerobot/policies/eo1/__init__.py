#!/usr/bin/env python

from alohamini_lerobot.policies import _load_export

_EXPORTS = {
    "EO1Config": (".configuration_eo1", "EO1Config"),
    "EO1Policy": (".modeling_eo1", "EO1Policy"),
    "make_eo1_pre_post_processors": (".processor_eo1", "make_eo1_pre_post_processors"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    return _load_export(__name__, _EXPORTS, name)
