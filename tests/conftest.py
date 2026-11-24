"""Shared test setup. On machines without a Metal GPU (e.g. virtualized CI runners),
fall back to the CPU device so the model-free suite still runs — every test here is
numerics-light and device-agnostic."""

import mlx.core as mx

if not mx.metal.is_available():
    mx.set_default_device(mx.cpu)


# MLX_DSPARK_TEST_THREADGROUP_CEILING=N: refuse any custom-kernel launch of more than N threads
# per threadgroup, in mlx's own words. That ceiling belongs to a GPU and a compiled kernel (an
# M4 Pro admits 1,024 everywhere; GitHub's runner and an M2 Pro have refused 1,024 and 768), so a
# launch that only this Mac admits is invisible here until CI runs it. One suite run under the
# lowest ceiling seen (384) before a release finds those: a failure is a code path without a
# fallback, or a kernel test without its skip.
import math
import os

_CEILING = int(os.environ.get("MLX_DSPARK_TEST_THREADGROUP_CEILING", "0") or 0)
if _CEILING and mx.metal.is_available():
    _metal_kernel = mx.fast.metal_kernel

    def _capped_metal_kernel(*args, **kwargs):
        kernel = _metal_kernel(*args, **kwargs)

        def launch(*a, **kw):
            threads = math.prod(int(x) for x in kw.get("threadgroup") or (0,))
            if threads > _CEILING:
                raise ValueError(f"Thread group size ({threads}) is greater than  the maximum "
                                 f"allowed threads per threadgroup ({_CEILING}).")
            return kernel(*a, **kw)

        return launch

    mx.fast.metal_kernel = _capped_metal_kernel
