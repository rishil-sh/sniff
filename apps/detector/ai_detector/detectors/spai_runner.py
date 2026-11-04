"""Runs inside SPAI's own venv (scripts/setup_spai.sh), cwd = the SPAI checkout.

Mirrors SPAI's `infer` path (CUDA-only upstream) on CPU/MPS: same config, model and
test transform; the checkpoint is loaded tensors-only. Prints one JSON line per image path in argv.
"""

import json
import pickle
import sys
import types

import numpy as np
import torch
from PIL import Image

# spai.models imports OpenAI `clip` at module level; only its unused CLIP backbone needs it.
sys.modules.setdefault("clip", types.ModuleType("clip"))

from spai.config import get_config  # noqa: E402 - needs the clip stub above
from spai.data.data_finetune import build_transform  # noqa: E402 - needs the clip stub above
from spai.models import build_cls_model  # noqa: E402 - needs the clip stub above


class _Unpickler(pickle.Unpickler):
    # torch 2.3 has no add_safe_globals; the checkpoint embeds a yacs config, so allow only tensors + that class.
    def find_class(self, module, name):
        if module.startswith("torch.") or (module, name) in {
            ("collections", "OrderedDict"),
            ("yacs.config", "CfgNode"),
            ("builtins", "set"),
            ("__builtin__", "set"),
        }:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


class _SafePickle:
    Unpickler = _Unpickler
    load = staticmethod(lambda f, **kw: _Unpickler(f, **kw).load())
    __name__ = "safepickle"


def main(paths: list[str]) -> None:
    config = get_config({"cfg": "configs/spai.yaml", "pretrained": "weights/spai.pth", "opts": []})
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_cls_model(config)
    # Upstream load_pretrained uses weights_only=False; tensors-only loading can't run pickled code.
    state = torch.load("weights/spai.pth", map_location="cpu", pickle_module=_SafePickle, weights_only=False)["model"]
    if any(k.startswith("encoder.") for k in state):
        state = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
    state = {k: v for k, v in state.items() if "relative_position_index" not in k}
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        sys.exit(f"SPAI checkpoint is missing {len(missing)} weights, e.g. {missing[:3]}")
    model.to(device).eval()
    transform = build_transform(is_train=False, config=config)
    for path in paths:
        image = np.array(Image.open(path).convert("RGB"))
        tensor = transform(image=image)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model([tensor], config.MODEL.FEATURE_EXTRACTION_BATCH).reshape(-1)[0].float().cpu()
        print(json.dumps({"path": path, "logit": float(logit), "p_ai": float(torch.sigmoid(logit))}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
