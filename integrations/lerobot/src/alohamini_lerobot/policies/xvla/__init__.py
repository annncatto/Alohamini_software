from alohamini_lerobot.policies import _load_export

_EXPORTS = {
    "XVLAConfig": (".configuration_xvla", "XVLAConfig"),
    "XVLAPolicy": (".modeling_xvla", "XVLAPolicy"),
    "XVLAAddDomainIdProcessorStep": (".processor_xvla", "XVLAAddDomainIdProcessorStep"),
    "XVLAImageNetNormalizeProcessorStep": (".processor_xvla", "XVLAImageNetNormalizeProcessorStep"),
    "XVLAImageToFloatProcessorStep": (".processor_xvla", "XVLAImageToFloatProcessorStep"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    return _load_export(__name__, _EXPORTS, name)
