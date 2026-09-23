from .adapter import QwenAdapter
from .layout import cached_img_shapes, layout_from_img_shapes, qwen_layout

__all__ = ["QwenAdapter", "cached_img_shapes", "layout_from_img_shapes", "qwen_layout"]

# Deliberately do not export shim here: it lazily imports diffusers' apply_rotary_emb_qwen, while
# the rest of models/ depends only on torch. `from anchorcache.models.qwen_image_edit import shim`
# remains supported. See models/flux2/__init__.py for the same rationale.

# model/processor is the built-in Qwen backend; at runtime it depends only on diffusers and model
# weights.
