import yaml
import time
import os
import sys
from typing import Any, Dict, Optional, List, Union


class AttributeDict:
    def __init__(self, data: Dict):
        object.__setattr__(self, '_data', data)

    def __getattr__(self, name: str) -> Any:
        value = self._data.get(name)

        if isinstance(value, dict):
            return AttributeDict(value)
        elif isinstance(value, list):
            return [AttributeDict(item) if isinstance(item, dict) else item for item in value]
        elif value is None and name not in self._data:
            return None
        else:
            return value

    def __getitem__(self, key: str) -> Any:
        try:
            value = self._data[key]
            if isinstance(value, dict):
                return AttributeDict(value)
            elif isinstance(value, list):
                return [AttributeDict(item) if isinstance(item, dict) else item for item in value]
            else:
                return value
        except KeyError:
            raise KeyError(key)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self._data)})"

    def __str__(self) -> str:
        return str(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()


class Config:
    _instance = None
    _config: Optional[Dict] = None

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.environ.get("CONFIG_PATH")
            if not config_path:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                default_path = os.path.join(script_dir, "..", "config", "config.yaml")
                print(f"CONFIG_PATH environment variable not set. Using default: {default_path}")
                config_path = default_path

        print(f"Loading configuration from: {config_path}")

        try:
            from yaml import CSafeLoader as SafeLoader
        except ImportError:
            from yaml import SafeLoader

        try:
            with open(config_path, 'r') as file:
                raw_config = yaml.load(file, Loader=SafeLoader)
                self._config = raw_config if isinstance(raw_config, dict) else {}
        except FileNotFoundError:
            print(f"Error: Config file not found at {config_path}")
            self._config = {}
            raise
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")
            self._config = {}
            raise
        except Exception as e:
            print(f"An unexpected error occurred during config loading: {e}")
            self._config = {}
            raise

    def __getattr__(self, name: str) -> Any:
        if self._config is None:
            raise AttributeError("Configuration not loaded.")

        value = self._config.get(name)

        if isinstance(value, dict):
            return AttributeDict(value)
        elif isinstance(value, list):
            return [AttributeDict(item) if isinstance(item, dict) else item for item in value]
        elif value is None and name not in self._config:
            return None
        else:
            return value

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        if self._config:
            return self._config.get(key, default)
        return default

    def __getitem__(self, key: str) -> Any:
        if self._config is None:
            raise KeyError("Configuration not loaded.")

        try:
            value = self._config[key]
            if isinstance(value, dict):
                return AttributeDict(value)
            elif isinstance(value, list):
                return [AttributeDict(item) if isinstance(item, dict) else item for item in value]
            else:
                return value
        except KeyError:
            raise KeyError(f"Key '{key}' not found in configuration.")

    @property
    def as_dict(self) -> Dict:
        return self._config.copy() if self._config else {}

    def reload(self, config_path: Optional[str] = None):
        print(f"Reloading configuration...")
        self._load_config(config_path)
        return self

    def __repr__(self) -> str:
        return f"Config({repr(self._config)})"


def get_config(config_path=None):
    try:
        config = Config(config_path)
        return config
    except Exception as e:
        print(f"Failed to get config instance: {e}")
        return None
