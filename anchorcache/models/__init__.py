# Backend adapters. Core must never import anything from this package; the dependency direction is one-way.
#
# This package deliberately provides no registry or automatic from_pretrained dispatch: the
# dependency stacks of the three backends are mutually incompatible (diffusers 0.39 versus
# transformers and flash-attn built against torch 2.5.1), and a single import can break the
# environment. Explicitly import each adapter only when needed.

__all__: list[str] = []
