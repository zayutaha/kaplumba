import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache

print("Loading model...")
model, _ = load('/Users/zayaantaha/.omlx/models/Qwwwon')
cache = make_prompt_cache(model, turbo_kv_bits=4)

print("Layer 0 state:")
print(cache[0].cache)
# Manually inject a sentinel and try to concatenate
try:
    conv_state = mx.zeros((1, 3, 10240), dtype=mx.float16)
    qkv_chunk = mx.zeros((1, 1, 10240), dtype=mx.float16)
    print("Shapes:", conv_state.shape, qkv_chunk.shape)
    res = mx.concatenate([conv_state, qkv_chunk], axis=1)
    print("Concatenate success!")
except Exception as e:
    print(f"Concatenate failed: {e}")
