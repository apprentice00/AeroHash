"""Full-model checkpoints, including exact legacy Python class locations."""
import importlib
import pickle
from types import ModuleType
import torch

_LEGACY_CLASSES = {
    **{('models.build_model', name): ('aerohash.models.encoders', name)
       for name in ('ECALayer', 'TSEncoder', 'FreqEncoder')},
    ('models.build_model', 'GatedFusion'): ('aerohash.models.fusion', 'GatedFusion'),
    ('models.build_model', 'MyModelTS'): ('aerohash.models.hashing', 'MyModelTS'),
    **{('models.Transformer', name): ('aerohash.models.transformer', name)
       for name in ('Transformer', 'TransformerEncoder', 'TransformerEncoderLayer')},
    ('models.center_loss', 'SeverityCenterLoss'): ('aerohash.losses.center', 'SeverityCenterLoss'),
}


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('models.Transformer', 'aerohash.models.transformer') and name in (
                'TransformerDecoder', 'TransformerDecoderLayer'):
            raise ValueError('Only GAP checkpoints are supported; query decoder checkpoints have been removed')
        target = _LEGACY_CLASSES.get((module, name))
        if target:
            return getattr(importlib.import_module(target[0]), target[1])
        return super().find_class(module, name)


_pickle = ModuleType('aerohash_checkpoint_pickle')
_pickle.Unpickler = LegacyUnpickler
_pickle.load = pickle.load
_pickle.loads = pickle.loads


def load_checkpoint(path, device='cpu'):
    """Load trusted full-model files without changing global module registrations."""
    model = torch.load(path, map_location=device, weights_only=False, pickle_module=_pickle)
    from aerohash.models.hashing import MyModelTS
    if isinstance(model, MyModelTS) and (
            not hasattr(model, 'mlp_head') or getattr(model, 'head_mode', 'gap_mlp') != 'gap_mlp'):
        raise ValueError('Only GAP checkpoints are supported; query/attention pooling checkpoints have been removed')
    return model


def save_checkpoint(model, path, device):
    """Preserve the original CPU serialization and return to the training device."""
    try:
        torch.save(model.cpu(), path)
    finally:
        model.to(device)
