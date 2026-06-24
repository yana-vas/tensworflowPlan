from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigDict(dict):

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(f"Config has no attribute '{name}'") from exc
        return ConfigDict(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


DEFAULTS: Dict[str, Any] = {
    "model": {
        "encoder": {
            "variant": "dinov2_vits14",
            "freeze": True,
            "use_lora": False,
            "lora_r": 16,
            "lora_alpha": 16,
        },
        "decoder": {
            "d_model": 384,
            "n_heads": 6,
            "n_layers": 4,
            "num_bands": 10,
            "dropout": 0.1,
            "ffn_mult": 4,
        },
    },
    "training": {
        "batch_size": 16,
        "num_points": 2048,
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "num_epochs": 100,
        "warmup_steps": 500,
        "grad_clip": 1.0,
        "amp": True,
        "num_workers": 4,
        "augment": True,
    },
    "inference": {
        "grid_resolution": 64,
        "threshold": 0.5,
        "query_batch_size": 100000,
    },
    "data": {
        "image_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    },
}


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> None:
    """Recursively merge ``update`` into ``base`` in place."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


class Config:

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None) -> None:
        import copy
        self._config = copy.deepcopy(DEFAULTS)
        if config_dict:
            _deep_update(self._config, config_dict)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        config = cls()
        yaml_path = Path(path)
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh)
            if loaded:
                _deep_update(config._config, loaded)
        return config

    @property
    def model(self) -> ConfigDict:
        return ConfigDict(self._config["model"])

    @property
    def training(self) -> ConfigDict:
        return ConfigDict(self._config["training"])

    @property
    def inference(self) -> ConfigDict:
        return ConfigDict(self._config["inference"])

    @property
    def data(self) -> ConfigDict:
        return ConfigDict(self._config["data"])

    def to_dict(self) -> Dict[str, Any]:
        import copy
        return copy.deepcopy(self._config)


def load_config(path: str = "configs/default.yaml") -> Config:
    return Config.from_yaml(path)


if __name__ == "__main__":
    cfg = load_config("configs/default.yaml")
    assert cfg.model.encoder.variant == "dinov2_vits14"
    assert cfg.model.decoder.d_model == 384
    assert cfg.model.decoder.d_model % cfg.model.decoder.n_heads == 0
    print("config.py self-test:", cfg.model.encoder.variant,
          "d_model", cfg.model.decoder.d_model, "heads", cfg.model.decoder.n_heads)