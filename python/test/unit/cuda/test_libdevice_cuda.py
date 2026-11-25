# fmt: off

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_cuda
from triton.language.extra import libdevice

pytestmark = pytest.mark.skipif(not is_cuda(), reason="Requires CUDA backend")


# -----------------------
# test extern functions
# -----------------------


@pytest.mark.parametrize("enable_reflect_ftz", [False, True])
def test_libdevice_precise_option_disables_ftz(enable_reflect_ftz, fresh_triton_cache):
    if not is_cuda():
        pytest.skip("requires CUDA")
    from triton.backends.nvidia.compiler import CUDAOptions
    assert CUDAOptions(enable_reflect_ftz=enable_reflect_ftz).enable_reflect_ftz == enable_reflect_ftz
    options = CUDAOptions(enable_reflect_ftz=enable_reflect_ftz, enable_precise_libdevice=True)
    assert not options.enable_reflect_ftz
    assert options.hash() == CUDAOptions(enable_precise_libdevice=True).hash()
    assert options.hash() != CUDAOptions(enable_reflect_ftz=False).hash()

    @triton.jit
    def kernel(X, Out):
        offsets = tl.arange(0, 4)
        x = tl.load(X + offsets)
        tl.store(Out + offsets, libdevice.sqrt(x))

    tiny = torch.finfo(torch.float32).tiny
    x = torch.tensor([tiny * torch.finfo(torch.float32).eps, tiny / 2, tiny, -0.], device="cuda")
    out = torch.empty_like(x)
    compiled = kernel[(1,)](x, out, enable_reflect_ftz=enable_reflect_ftz, enable_precise_libdevice=True)
    assert not compiled.metadata.enable_reflect_ftz
    assert compiled.metadata.enable_precise_libdevice
    assert "sqrt.rn.f32" in compiled.asm["ptx"]
    assert torch.equal(out.view(torch.int32), torch.sqrt(x).view(torch.int32))


@pytest.mark.parametrize("explicit_precise", [None, False, True])
def test_libdevice_precise_option_accepts_legacy_name(explicit_precise, fresh_triton_cache):
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend

    backend = CUDABackend(GPUTarget("cuda", 90, 32))
    options = {"enable_reflect_prec_sqrt": True}
    if explicit_precise is not None:
        options["enable_precise_libdevice"] = explicit_precise
    expected_precise = explicit_precise is not False
    parsed = backend.parse_options(options)
    canonical = backend.parse_options({"enable_precise_libdevice": expected_precise})
    assert parsed == canonical
    assert parsed.hash() == canonical.hash()
    assert backend.parse_options({"enable_reflect_prec_sqrt": False}) == backend.parse_options({})

    @triton.jit
    def kernel(X, Out):
        offsets = tl.arange(0, 4)
        tl.store(Out + offsets, libdevice.sqrt(tl.load(X + offsets)))

    x = torch.tensor([2., 3., 5., 7.], device="cuda")
    out = torch.empty_like(x)
    compiled = kernel[(1,)](x, out, **options)
    assert compiled.metadata.enable_precise_libdevice == expected_precise
    assert ("sqrt.rn.f32" in compiled.asm["ptx"]) == expected_precise
    if expected_precise:
        assert torch.equal(out.view(torch.int32), torch.sqrt(x).view(torch.int32))


@pytest.mark.parametrize("rounding", ["rna", "rn"])
def test_cuda_round_f32_to_tf32(rounding):
    capability = (9, 0) if rounding == "rn" else (8, 0)
    if not is_cuda() or torch.cuda.get_device_capability() < capability:
        pytest.skip(f"requires CUDA compute capability {capability}")

    @triton.jit
    def kernel(X, Out, ROUNDING: tl.constexpr):
        offsets = tl.arange(0, 4)
        x = tl.load(X + offsets)
        tl.store(Out + offsets, tl.extra.cuda.round_f32_to_tf32(x, ROUNDING))

    x = torch.tensor([1 + 2**-11, 1 + 3 * 2**-11, -1 - 2**-11, -1 - 3 * 2**-11], device="cuda")
    out = torch.empty_like(x)
    kernel[(1,)](x, out, rounding)
    rounded = 1. if rounding == "rn" else 1 + 2**-10
    expected = torch.tensor([rounded, 1 + 2**-9, -rounded, -1 - 2**-9], device="cuda")
    assert torch.equal(out.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("propagate_nan", [tl.PropagateNan.NONE, tl.PropagateNan.ALL])
def test_cuda_minmax_xorsign_abs(op, propagate_nan):
    if not is_cuda() or torch.cuda.get_device_capability() < (8, 6):
        pytest.skip("requires CUDA compute capability 8.6+")

    @triton.jit
    def kernel(A, B, Out, OP: tl.constexpr, PROPAGATE_NAN: tl.constexpr):
        offsets = tl.arange(0, 8)
        a = tl.load(A + offsets)
        b = tl.load(B + offsets)
        if OP == "min":
            result = tl.extra.cuda.min_xorsign_abs_f32(a, b, propagate_nan=PROPAGATE_NAN)
        else:
            result = tl.extra.cuda.max_xorsign_abs_f32(a, b, propagate_nan=PROPAGATE_NAN)
        tl.store(Out + offsets, result)

    a = torch.tensor([2., -2., 2., -2., float("nan"), 1., -0., 0.], device="cuda")
    b = torch.tensor([3., 3., -3., -3., 1., float("nan"), 0., -0.], device="cuda")
    out = torch.empty_like(a)
    kernel[(1,)](a, b, out, op, propagate_nan)
    magnitude = 2. if op == "min" else 3.
    nan_result = float("nan") if propagate_nan == tl.PropagateNan.ALL else 1.
    expected = torch.tensor([magnitude, -magnitude, -magnitude, magnitude, nan_result, nan_result, -0., -0.],
                            device="cuda")
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(out[6:]), torch.signbit(expected[6:]))


@pytest.mark.parametrize("enable_fp_fusion", [False, True])
def test_libdevice_sqrt_precision_reflection(enable_fp_fusion, fresh_knobs, fresh_triton_cache):
    if not is_cuda():
        pytest.skip("requires CUDA")
    fresh_knobs.language.allow_fp_fusion = True

    @triton.jit
    def kernel(X, Out, N: tl.constexpr, TAG: tl.constexpr):
        offsets = tl.arange(0, 1024)
        x = tl.load(X + offsets, offsets < N)
        tl.store(Out + offsets, libdevice.sqrt(x), offsets < N)

    generator = torch.Generator(device="cuda").manual_seed(42)
    x = torch.rand(1024, generator=generator, device="cuda") * 100
    tiny = torch.finfo(torch.float32).tiny
    min_subnormal = tiny * torch.finfo(torch.float32).eps
    x[:6] = torch.tensor([0., -0., tiny, min_subnormal, tiny / 2, float("inf")], device="cuda")
    out = torch.empty_like(x)
    expected = torch.sqrt(x)
    metadata = {}
    default_hash = None
    # Compile defaults again after the precise variant to check that the LLVM
    # reflection setting does not escape its compilation scope. TAG prevents
    # these checks from reusing the JIT cache instead of translating again.
    for tag, precise in enumerate([None, False, True, None, False]):
        options = {"enable_fp_fusion": enable_fp_fusion, "enable_reflect_ftz": False}
        if precise is not None:
            options["enable_precise_libdevice"] = precise
        compiled = kernel[(1,)](x, out, x.numel(), tag, **options)
        assert compiled.metadata.enable_fp_fusion == enable_fp_fusion
        if tag == 0:
            default_hash = compiled.hash
        metadata[precise] = compiled.metadata
        ptx = compiled.asm["ptx"]
        if precise:
            assert "sqrt.rn.f32" in ptx
            assert "sqrt.approx.f32" not in ptx
            assert torch.equal(out.view(torch.int32), expected.view(torch.int32))
        else:
            assert "sqrt.approx.f32" in ptx
            assert "sqrt.rn.f32" not in ptx

    from triton.backends.nvidia.compiler import CUDAOptions
    assert CUDAOptions().hash() == CUDAOptions(enable_precise_libdevice=False).hash()
    assert CUDAOptions().hash() != CUDAOptions(enable_precise_libdevice=True).hash()
    assert not metadata[None].enable_precise_libdevice
    assert not metadata[False].enable_precise_libdevice
    assert metadata[True].enable_precise_libdevice

    # The same source and specialization must select different binary-cache
    # entries when precision changes, and retain the existing default entry.
    common = {"enable_fp_fusion": enable_fp_fusion, "enable_reflect_ftz": False}
    approximate = kernel.warmup(x, out, x.numel(), 0, grid=(1,), enable_precise_libdevice=False, **common)
    precise = kernel.warmup(x, out, x.numel(), 0, grid=(1,), enable_precise_libdevice=True, **common)
    restored = kernel.warmup(x, out, x.numel(), 0, grid=(1,), **common)
    assert approximate.hash == default_hash == restored.hash
    assert approximate.hash != precise.hash

    # Exercise opposite native reflection settings concurrently without the
    # JIT or binary cache hiding the option registry's compilation scopes.
    from triton.backends.nvidia.compiler import CUDABackend, llvm
    # JIT calls above can reuse cached binaries and skip make_llir's target
    # initialization. Direct backend stage calls must initialize it explicitly.
    llvm.init_targets()
    target = triton.runtime.driver.active.get_current_target()
    backend = CUDABackend(target)

    def translate(precise):
        options = backend.parse_options({**common, "enable_precise_libdevice": precise})
        assert options.enable_fp_fusion == enable_fp_fusion
        ptx = backend.make_ptx(approximate.asm["llir"], {}, options, target.arch)
        assert ("sqrt.rn.f32" in ptx) == precise
        assert ("sqrt.approx.f32" in ptx) != precise

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(translate, [False, True] * 4))


@pytest.mark.parametrize("enable_fp_fusion", [False, True])
def test_libdevice_fp_fusion(enable_fp_fusion, fresh_knobs, fresh_triton_cache):
    if not is_cuda():
        pytest.skip("requires CUDA")
    fresh_knobs.language.allow_fp_fusion = True

    @triton.jit
    def kernel(X, E, A, B, C, Power, Arithmetic, N: tl.constexpr, TAG: tl.constexpr):
        offsets = tl.arange(0, 128)
        mask = offsets < N
        x = tl.load(X + offsets, mask, other=1)
        exponent = tl.load(E + offsets, mask, other=1)
        a = tl.load(A + offsets, mask, other=0)
        b = tl.load(B + offsets, mask, other=0)
        c = tl.load(C + offsets, mask, other=0)
        tl.store(Power + offsets, libdevice.pow(x, exponent), mask)
        tl.store(Arithmetic + offsets, a * b + c, mask)

    # These promoted bf16/fp16 inputs expose contraction inside libdevice's
    # compensated double-precision power routine.
    x = torch.tensor([2.2870594307278225e-14, 0.04119873046875], device="cuda", dtype=torch.float64)
    exponent = torch.tensor([8.4375, -107.5625], device="cuda", dtype=torch.float64)
    a = torch.full_like(x, 1 + 2**-27)
    b = torch.full_like(x, 1 - 2**-27)
    c = torch.full_like(x, -1)
    expected = torch.pow(x, exponent)
    power, arithmetic = torch.empty_like(x), torch.empty_like(x)
    expected_arithmetic = torch.full_like(x, -2**-54) if enable_fp_fusion else torch.zeros_like(x)
    common = {"enable_fp_fusion": enable_fp_fusion, "enable_reflect_ftz": False}
    args = (x, exponent, a, b, c, power, arithmetic, x.numel())
    default_bits = None
    for tag, external_fusion in enumerate([None, False, True, None, False]):
        options = dict(common)
        if external_fusion is not None:
            options["enable_libdevice_fp_fusion"] = external_fusion
        compiled = kernel[(1,)](*args, tag, **options)
        assert compiled.metadata.enable_libdevice_fp_fusion == bool(external_fusion)
        assert compiled.metadata.enable_fp_fusion == enable_fp_fusion
        assert torch.equal(arithmetic.view(torch.int64), expected_arithmetic.view(torch.int64))
        if external_fusion or enable_fp_fusion:
            assert torch.equal(power.view(torch.int64), expected.view(torch.int64))
        if tag == 0:
            default_bits = power.view(torch.int64).clone()
        elif not external_fusion:
            assert torch.equal(power.view(torch.int64), default_bits)

    from triton.backends.nvidia.compiler import CUDAOptions
    assert CUDAOptions().hash() == CUDAOptions(enable_libdevice_fp_fusion=False).hash()
    assert CUDAOptions().hash() != CUDAOptions(enable_libdevice_fp_fusion=True).hash()
    default = kernel.warmup(*args, 0, grid=(1,), **common)
    enabled = kernel.warmup(*args, 0, grid=(1,), enable_libdevice_fp_fusion=True, **common)
    restored = kernel.warmup(*args, 0, grid=(1,), **common)
    assert default.hash == restored.hash
    assert default.hash != enabled.hash


@pytest.mark.parametrize("roots", [(), ("__nv_pow",)])
def test_libdevice_fp64_fusion_preserves_float32(roots, fresh_knobs, fresh_triton_cache):
    if not is_cuda():
        pytest.skip("requires CUDA")
    fresh_knobs.language.allow_fp_fusion = True

    @triton.jit
    def kernel(X32, E32, X64, E64, Y32, Y64, TAG: tl.constexpr):
        offsets = tl.arange(0, 128)
        x32 = tl.load(X32 + offsets)
        e32 = tl.load(E32 + offsets)
        x64 = tl.load(X64 + offsets, offsets < 2, other=1)
        e64 = tl.load(E64 + offsets, offsets < 2, other=1)
        tl.store(Y32 + offsets, libdevice.pow(x32, e32))
        tl.store(Y64 + offsets, libdevice.pow(x64, e64), offsets < 2)

    x32 = torch.linspace(0.125, 8, 128, device="cuda")
    e32 = torch.linspace(-9, 9, 128, device="cuda")
    x64 = torch.tensor([2.2870594307278225e-14, 0.04119873046875], device="cuda", dtype=torch.float64)
    e64 = torch.tensor([8.4375, -107.5625], device="cuda", dtype=torch.float64)
    y32, y64 = torch.empty_like(x32), torch.empty_like(x64)
    expected64 = torch.pow(x64, e64)
    common = {"enable_fp_fusion": False, "enable_reflect_ftz": False, "libdevice_fp64_fusion_roots": roots}
    baseline32, baseline64 = None, None
    for tag, mode in enumerate([None, False, True, None, False]):
        options = dict(common)
        if mode is not None:
            options["enable_libdevice_fp64_fusion"] = mode
        compiled = kernel[(1,)](x32, e32, x64, e64, y32, y64, tag, **options)
        assert compiled.metadata.enable_libdevice_fp64_fusion == bool(mode)
        assert tuple(compiled.metadata.libdevice_fp64_fusion_roots) == roots
        assert not compiled.metadata.enable_libdevice_fp_fusion
        assert not compiled.metadata.enable_fp_fusion
        if tag == 0:
            baseline32, baseline64 = y32.view(torch.int32).clone(), y64.view(torch.int64).clone()
        assert torch.equal(y32.view(torch.int32), baseline32)
        if mode:
            assert torch.equal(y64.view(torch.int64), expected64.view(torch.int64))
        else:
            assert torch.equal(y64.view(torch.int64), baseline64)

    from triton.backends.nvidia.compiler import CUDAOptions
    assert CUDAOptions().hash() == CUDAOptions(enable_libdevice_fp64_fusion=False).hash()
    assert CUDAOptions().hash() != CUDAOptions(enable_libdevice_fp64_fusion=True).hash()
    assert CUDAOptions(libdevice_fp64_fusion_roots=list(roots)).libdevice_fp64_fusion_roots == roots
    assert CUDAOptions().hash() != CUDAOptions(libdevice_fp64_fusion_roots=("__nv_pow",)).hash()


@pytest.mark.parametrize("with_power", [False, True])
@pytest.mark.parametrize("negative_gamma", [False, True])
def test_libdevice_fp64_fusion_roots_preserve_other_functions(with_power, negative_gamma, fresh_knobs, fresh_triton_cache):
    if not is_cuda():
        pytest.skip("requires CUDA")
    fresh_knobs.language.allow_fp_fusion = True

    @triton.jit
    def kernel(X, E, GammaInput, Power, Logaddexp, Gamma, WITH_POWER: tl.constexpr, TAG: tl.constexpr):
        offsets = tl.arange(0, 128)
        x = tl.load(X + offsets, offsets < 2, other=1)
        exponent = tl.load(E + offsets, offsets < 2, other=1)
        if WITH_POWER:
            tl.store(Power + offsets, libdevice.pow(x, exponent), offsets < 2)
        a = tl.full((128,), 0.5, tl.float64)
        b = tl.full((128,), 1.0000001192092896, tl.float64)
        tl.store(Logaddexp + offsets, b + libdevice.log1p(libdevice.exp(a - b)))
        gamma_input = tl.load(GammaInput + offsets)
        tl.store(Gamma + offsets, libdevice.tgamma(gamma_input))

    x = torch.tensor([2.2870594307278225e-14, 0.04119873046875], device="cuda", dtype=torch.float64)
    exponent = torch.tensor([8.4375, -107.5625], device="cuda", dtype=torch.float64)
    gamma_input = torch.linspace(0.01, 170, 128, device="cuda", dtype=torch.float64)
    if negative_gamma:
        gamma_input = torch.full_like(gamma_input, -99.36625884608715)
    power, logaddexp, gamma = torch.empty_like(x), torch.empty_like(gamma_input), torch.empty_like(gamma_input)
    expected_power = torch.pow(x, exponent).view(torch.int64)
    expected_logaddexp = torch.logaddexp(torch.full_like(logaddexp, 0.5), torch.full_like(logaddexp, 1.0000001192092896))
    args = (x, exponent, gamma_input, power, logaddexp, gamma, with_power)
    common = {"enable_fp_fusion": False, "enable_reflect_ftz": False}
    baseline_gamma, broad_results = None, None
    modes = [({}, None), ({"enable_libdevice_fp64_fusion": True, "libdevice_fp64_fusion_roots": ("__nv_pow",)}, True),
             ({}, None), ({"enable_libdevice_fp_fusion": True}, False),
             ({"enable_libdevice_fp_fusion": True, "libdevice_fp64_fusion_roots": ("__nv_pow",)}, False)]
    for tag, (options, scoped) in enumerate(modes):
        kernel[(1,)](*args, tag, **common, **options)
        if tag == 0:
            baseline_gamma = gamma.view(torch.int64).clone()
        if scoped is not False:
            assert torch.equal(logaddexp.view(torch.int64), expected_logaddexp.view(torch.int64))
            assert torch.equal(gamma.view(torch.int64), baseline_gamma)
        if scoped and with_power:
            assert torch.equal(power.view(torch.int64), expected_power)
        if tag == 3:
            broad_results = (logaddexp.view(torch.int64).clone(), gamma.view(torch.int64).clone())
        elif tag == 4:
            assert torch.equal(logaddexp.view(torch.int64), broad_results[0])
            assert torch.equal(gamma.view(torch.int64), broad_results[1])

    default = kernel.warmup(*args, 0, grid=(1,), **common)
    selected = kernel.warmup(*args, 0, grid=(1,), **common, **modes[1][0])
    restored = kernel.warmup(*args, 0, grid=(1,), **common)
    assert default.hash == restored.hash
    assert default.hash != selected.hash
    with pytest.raises(Exception, match="Unknown external-library contraction root"):
        kernel.warmup(*args, 0, grid=(1,), **common, enable_libdevice_fp64_fusion=True,
                      libdevice_fp64_fusion_roots=("__nv_missing_function",))


@triton.jit
def tanh_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    direct_import: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    if direct_import:
        y = libdevice.tanh(x)
    else:
        y = tl.extra.libdevice.tanh(x)

    tl.store(y_ptr + offsets, y, mask=mask)

def test_cuda_f32_to_e2m1x2_satfinite():
    if not is_cuda() or torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("requires Blackwell")

    @triton.jit
    def kernel(Hi, Lo, Out):
        offsets = tl.arange(0, 32)
        hi = tl.load(Hi + offsets)
        lo = tl.load(Lo + offsets)
        tl.store(Out + offsets, tl.extra.cuda.f32_to_e2m1x2_satfinite(hi, lo))

    # Exact values, ties to even, saturation, and both signs of zero.
    values = torch.tensor([0., .25, .5, .75, 1., 1.25, 1.5, 1.75,
                           2., 2.5, 3., 3.5, 4., 5., 6., float("inf")], device="cuda")
    codes = torch.tensor([0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 7], device="cuda")
    hi = torch.cat((values, -values))
    codes = torch.cat((codes, codes | 8))
    lo = hi.flip(0)
    out = torch.empty(32, dtype=torch.uint16, device="cuda")
    h = kernel[(1,)](hi, lo, out)
    expected = (codes << 4) | codes.flip(0)
    assert torch.equal(out.to(torch.int64), expected)
    assert "cvt.rn.satfinite.e2m1x2.f32" in h.asm["ptx"]
    assert ".pzo" not in h.asm["ptx"]


@pytest.mark.parametrize("direct_import", [False, True])
@pytest.mark.parametrize("dtype_str", ['float32', 'float64'])
def test_math_extern(dtype_str, direct_import):

    if not torch.cuda.is_available():
        pytest.skip("Test requires CUDA target.")
        return

    torch.manual_seed(42)

    x = torch.randn((100,), dtype=getattr(torch, dtype_str), device="cuda")

    y_tri = torch.empty_like(x)
    tanh_kernel[(1, )](x, y_tri, x.shape[0], direct_import, BLOCK_SIZE=128)

    y_ref = torch.tanh(x)
    np.testing.assert_allclose(y_ref.cpu().numpy(), y_tri.cpu().numpy(), rtol=0, atol=1.0e-6)


@triton.jit
def builtin_libdevice_unary_kernel(
    x_ptr,
    builtin_ptr,
    libdevice_ptr,
    n_elements,
    OP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    if OP == "exp":
        builtin = tl.exp(x)
        external = libdevice.exp(x)
    elif OP == "fast_exp":
        builtin = tl.exp(x)
        external = libdevice.fast_expf(x)
    elif OP == "sqrt_rn":
        builtin = tl.sqrt_rn(x)
        external = libdevice.sqrt_rn(x)
    elif OP == "exp2":
        builtin = tl.exp2(x)
        external = libdevice.exp2(x)
    elif OP == "log":
        builtin = tl.log(x)
        external = libdevice.log(x)
    elif OP == "log2":
        builtin = tl.log2(x)
        external = libdevice.log2(x)
    elif OP == "sin":
        builtin = tl.sin(x)
        external = libdevice.sin(x)
    elif OP == "cos":
        builtin = tl.cos(x)
        external = libdevice.cos(x)
    elif OP == "rsqrt":
        builtin = tl.rsqrt(x)
        external = libdevice.rsqrt(x)
    elif OP == "erf":
        builtin = tl.erf(x)
        external = libdevice.erf(x)
    elif OP == "floor":
        builtin = tl.floor(x)
        external = libdevice.floor(x)
    elif OP == "ceil":
        builtin = tl.ceil(x)
        external = libdevice.ceil(x)
    elif OP == "fast_sin":
        builtin = tl.sin(x)
        external = libdevice.fast_sinf(x)
    elif OP == "fast_cos":
        builtin = tl.cos(x)
        external = libdevice.fast_cosf(x)
    elif OP == "fast_log":
        builtin = tl.log(x)
        external = libdevice.fast_logf(x)
    elif OP == "fast_log2":
        builtin = tl.log2(x)
        external = libdevice.fast_log2f(x)

    tl.store(builtin_ptr + offsets, builtin, mask=mask)
    tl.store(libdevice_ptr + offsets, external, mask=mask)


@triton.jit
def builtin_libdevice_div_rn_kernel(
    x_ptr,
    y_ptr,
    builtin_ptr,
    libdevice_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    tl.store(builtin_ptr + offsets, tl.div_rn(x, y), mask=mask)
    tl.store(libdevice_ptr + offsets, libdevice.div_rn(x, y), mask=mask)


def _random_payloads(generator, n_elements):
    return torch.randint(-(2**31), 2**31 - 1, (n_elements,), dtype=torch.int32, device="cuda", generator=generator)


@pytest.mark.parametrize(
    "op", ["exp", "fast_exp", "exp2", "log", "log2", "sin", "cos", "sqrt_rn", "rsqrt", "erf", "floor", "ceil"]
)
def test_fpsan_libdevice_unary_equivalence(op, fresh_knobs):
    if not is_cuda() or not torch.cuda.is_available():
        pytest.skip("test requires CUDA")

    fresh_knobs.compilation.instrumentation_mode = "fpsan"
    n_elements = 1024
    block_size = 256
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    x = _random_payloads(generator, n_elements)
    builtin = torch.empty_like(x)
    external = torch.empty_like(x)

    builtin_libdevice_unary_kernel[(triton.cdiv(n_elements, block_size),)](
        triton.TensorWrapper(x, dtype=torch.float32),
        triton.TensorWrapper(builtin, dtype=torch.float32),
        triton.TensorWrapper(external, dtype=torch.float32),
        n_elements,
        OP=op,
        BLOCK_SIZE=block_size,
    )

    assert torch.equal(builtin, external)


def test_fpsan_libdevice_div_rn_equivalence(fresh_knobs):
    if not is_cuda() or not torch.cuda.is_available():
        pytest.skip("test requires CUDA")

    fresh_knobs.compilation.instrumentation_mode = "fpsan"
    n_elements = 1024
    block_size = 256
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    inputs = [_random_payloads(generator, n_elements) for _ in range(2)]
    builtin = torch.empty_like(inputs[0])
    external = torch.empty_like(inputs[0])

    builtin_libdevice_div_rn_kernel[(triton.cdiv(n_elements, block_size),)](
        *(triton.TensorWrapper(value, dtype=torch.float32) for value in inputs),
        triton.TensorWrapper(builtin, dtype=torch.float32),
        triton.TensorWrapper(external, dtype=torch.float32),
        n_elements,
        BLOCK_SIZE=block_size,
    )

    assert torch.equal(builtin, external)


@pytest.mark.parametrize("op", ["fast_sin", "fast_cos", "fast_log", "fast_log2"])
def test_fpsan_distinguishes_approximate_libdevice(op, fresh_knobs):
    if not is_cuda() or not torch.cuda.is_available():
        pytest.skip("test requires CUDA")

    fresh_knobs.compilation.instrumentation_mode = "fpsan"
    n_elements = 1024
    block_size = 256
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    x = _random_payloads(generator, n_elements)
    builtin = torch.empty_like(x)
    external = torch.empty_like(x)

    builtin_libdevice_unary_kernel[(triton.cdiv(n_elements, block_size),)](
        triton.TensorWrapper(x, dtype=torch.float32),
        triton.TensorWrapper(builtin, dtype=torch.float32),
        triton.TensorWrapper(external, dtype=torch.float32),
        n_elements,
        OP=op,
        BLOCK_SIZE=block_size,
    )

    assert not torch.equal(builtin, external)
