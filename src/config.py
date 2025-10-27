import yaml
import os
import ast
from typing import Any, Dict, List, Union


class Config(dict):
    def __init__(self, config_path: str = '', data: Dict[str, Any] = None):
        """
        Initialize from a YAML file or directly from a dict.
        """
        if data is None:
            with open(config_path, 'r') as f:
                self._yaml = f.read()
                self._dict = yaml.safe_load(self._yaml)
                self._dict['PATH'] = os.path.dirname(config_path)
        else:
            self._dict = dict(data)
            # Best-effort PATH if a file was provided; else CWD
            self._dict['PATH'] = self._dict.get('PATH', os.getcwd())
            self._yaml = yaml.safe_dump(self._dict, sort_keys=False)

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]
        return None

    def set(self, key: str, value: Any) -> None:
        self._dict[key] = value
        # refresh cached yaml
        self._yaml = yaml.safe_dump(self._dict, sort_keys=False)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._dict)

    def to_yaml(self) -> str:
        # Use safe_dump to render effective config
        return yaml.safe_dump(self._dict, sort_keys=False)

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')
def _parse_cli_value(raw: str) -> Any:
    s = raw.strip()
    # Lists and dicts via literal_eval when obvious
    if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')) or (s.startswith('{') and s.endswith('}')):
        try:
            return ast.literal_eval(s)
        except Exception:
            pass
    # Bool as string to remain compatible with existing 'True'/'False' checks
    lower = s.lower()
    if lower in ('true', 'false'):
        return 'True' if lower == 'true' else 'False'
    # Int / float
    try:
        if s.startswith('0') and s != '0' and not s.startswith('0.'):
            # keep as string to avoid octal-like confusion
            return s
        iv = int(s)
        return iv
    except ValueError:
        try:
            fv = float(s)
            return fv
        except ValueError:
            return s


def _apply_overrides(base: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    if not overrides:
        return base
    merged = dict(base)
    for item in overrides:
        if not item:
            continue
        if '=' not in item:
            # ignore silently; or could raise
            continue
        k, v = item.split('=', 1)
        k = k.strip()
        v_parsed = _parse_cli_value(v)
        merged[k] = v_parsed
    return merged


def load_config(path: Union[str, List[str]], overrides: List[str] = None) -> Config:
    """
    Load a config from a YAML file (or list of files), applying CLI overrides.
    - If multiple files are given, later files override earlier ones (shallow merge).
    - Overrides are given as KEY=VALUE strings (repeatable).
    """
    paths: List[str]
    if isinstance(path, list):
        paths = path
    else:
        # support comma-separated list in a single arg
        paths = [p for p in str(path).split(',') if p.strip()]
    # Start with first
    merged: Dict[str, Any] = {}
    base_dir = os.getcwd()
    for i, p in enumerate(paths):
        p = p.strip()
        with open(p, 'r') as f:
            d = yaml.safe_load(f)
            merged.update(d or {})
        if i == 0:
            base_dir = os.path.dirname(p)
    # Apply CLI overrides
    merged = _apply_overrides(merged, overrides or [])
    # Set PATH to the directory of the first config for reproducibility
    merged['PATH'] = base_dir
    return Config(data=merged)
