"""Resolve typed configuration with explicit CLI > env > YAML > defaults precedence."""
from __future__ import annotations
import argparse
import json
import os
import types
from dataclasses import asdict, fields
from importlib.resources import files
from pathlib import Path
from typing import get_args, get_origin, get_type_hints
import yaml
from .schema import ExperimentConfig
from .environment import ENVIRONMENT, REMOVED_MODEL_ENVIRONMENT
from aerohash.paths import PROJECT_ROOT


def _convert(value, hint, key):
    origin = get_origin(hint)
    if origin is types.UnionType:
        if value is None or (isinstance(value, str) and value.lower() in ('none', 'null')):
            return None
        return _convert(value, next(t for t in get_args(hint) if t is not type(None)), key)
    if origin is list:
        item_hint = get_args(hint)[0]
        if isinstance(value, str):
            value = [s.strip() for s in value.split(',') if s.strip()]
        if not isinstance(value, list):
            raise ValueError(f'{key}: expected a list')
        items = [part.strip() if isinstance(part, str) else part for item in value for part in (item.split(',') if isinstance(item, str) else [item]) if not isinstance(part, str) or part.strip()]
        return [_convert(item, item_hint, key) for item in items]
    if hint is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ('1', '0', 'true', 'false'):
            return value.lower() in ('1', 'true')
        raise ValueError(f'{key}: expected true/false or 1/0')
    if hint is str:
        if not isinstance(value, str):
            raise ValueError(f'{key}: expected a string')
        return value
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f'{key}: expected {hint.__name__}')
    if hint is int and isinstance(value, float) and not value.is_integer():
        raise ValueError(f'{key}: expected an integer')
    try:
        return hint(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f'{key}: invalid {hint.__name__}: {value!r}') from e


def _apply(config, values):
    if not isinstance(values, dict):
        raise ValueError('Configuration must be a mapping of sections')
    sections = {f.name for f in fields(config)}
    for group, options in values.items():
        if group not in sections:
            raise ValueError(f'Unknown configuration section: {group}')
        if not isinstance(options, dict):
            raise ValueError(f'{group}: expected a mapping')
        section = getattr(config, group)
        hints = get_type_hints(type(section))
        for name, value in options.items():
            if name not in hints:
                raise ValueError(f'Unknown configuration key: {group}.{name}')
            setattr(section, name, _convert(value, hints[name], f'{group}.{name}'))


def load_config(path=None, overrides=None, environ=None):
    config = ExperimentConfig()
    _apply(config, yaml.safe_load(files(__package__).joinpath('default.yaml').read_text()))
    if path is not None:
        _apply(config, yaml.safe_load(Path(path).read_text(encoding='utf-8')))
    environ = os.environ if environ is None else environ
    removed = [key for key in REMOVED_MODEL_ENVIRONMENT if key in environ]
    if removed:
        raise ValueError(f'GAP is the only readout; remove obsolete environment variables: {", ".join(removed)}')
    env_values = {}
    for variable, key in ENVIRONMENT.items():
        if variable in environ:
            group, name = key.split('.')
            value = environ[variable]
            # Historical boolean switches use exactly '1', not Python truthiness.
            if isinstance(getattr(getattr(config, group), name), bool):
                value = value == '1'
            env_values.setdefault(group, {})[name] = value
    _apply(config, env_values)
    cli_values = {}
    for key, value in (overrides or {}).items():
        group, name = key.split('.')
        cli_values.setdefault(group, {})[name] = value
    _apply(config, cli_values)
    if not config.data.stations:
        raise ValueError('data.stations must contain at least one station')
    for name in ('code_length', 'lr_step'):
        if not getattr(config.training, name):
            raise ValueError(f'training.{name} must not be empty')
    return config


def configuration_parser(command):
    """Expose all section.field options and convenient command-specific aliases."""
    parser = argparse.ArgumentParser(prog=f'aerohash {command}')
    parser.add_argument('--config', help='YAML overrides to the packaged default experiment')
    config = ExperimentConfig()
    active = 'training' if command == 'train' else 'evaluation'
    aliases = {'data.stations': '--stations', 'data.dataset_root': '--data-root',
               'output.ckpt_root': '--ckpt-root', 'output.ckpt_name': '--ckpt-name'}
    for group_field in fields(config):
        group = group_field.name
        section = getattr(config, group)
        for item in fields(section):
            key = f'{group}.{item.name}'
            flags = ['--'+key.replace('_', '-')]
            alias = aliases.get(key)
            if group == active:
                alias = '--'+item.name.replace('_', '-')
            if alias and alias not in flags:
                flags.append(alias)
            kwargs = {'nargs': '+'} if isinstance(getattr(section, item.name), list) else {}
            parser.add_argument(*flags, **kwargs, dest=key, default=argparse.SUPPRESS,
                                metavar='VALUE', help=f'{key} (default: {getattr(section, item.name)!r})')
    return parser


def parse_config(command, argv=None):
    parser = configuration_parser(command)
    values = vars(parser.parse_args(argv))
    path = values.pop('config', None)
    try:
        return load_config(path, values)
    except (ValueError, OSError, yaml.YAMLError) as e:
        parser.error(str(e))


def station_roots(config):
    root = Path(config.data.dataset_root) if config.data.dataset_root is not None else PROJECT_ROOT/'dataset'
    return [os.path.normpath(root/station) for station in config.data.stations]


def training_namespace(config):
    """Adapt grouped public configuration to the existing numerical kernel arguments.

    Derived dimensions always come from station metadata, as in the original code.
    Evaluation CLI fields are not flattened, to avoid batch/device name collisions.
    """
    values = {}
    for group in ('data', 'model', 'training', 'loss', 'output'):
        values.update(asdict(getattr(config, group)))
    values['use_val_select'] = config.evaluation.use_val_select
    values['roots'] = station_roots(config)
    values['root'] = values['roots'][0]
    with open(Path(values['root'])/'config.json', encoding='utf-8') as stream:
        metadata = json.load(stream)
    values.update(in_dim=int(metadata['D']), seq_len=int(metadata['window']), num_classes=int(metadata['K']))
    import torch
    values['device'] = torch.device(config.training.device or ('cpu' if config.training.gpu is None else f'cuda:{config.training.gpu}'))
    values['_config'] = asdict(config)
    return argparse.Namespace(**values)


def write_resolved_config(args, directory):
    """Record requested sections and effective dimensions/paths for this run."""
    content = dict(args._config)
    content['runtime'] = {k: str(v) if k == 'device' else v for k, v in vars(args).items()
                          if not k.startswith('_')}
    Path(directory, 'resolved_config.yaml').write_text(yaml.safe_dump(content, sort_keys=False), encoding='utf-8')
