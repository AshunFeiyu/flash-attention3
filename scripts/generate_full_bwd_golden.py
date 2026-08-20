#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np


SCHEMA = "shaobo-fa3-bwd-golden-v1"
FORMULA_VERSION = "deterministic-bhsd-fp16-v1"
GQA_SCHEMA = "shaobo-fa3-bwd-golden-v2-gqa"
GQA_FORMULA_VERSION = "deterministic-bhsd-fp16-gqa-v1"
LOG2_E = np.float32(1.4426950408889634)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or validate a cached CPU golden for full FA BWD."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--heads-kv", type=int)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--causal", type=int, choices=(0, 1), default=1)
    parser.add_argument("--softmax-scale", type=float, default=0.08838834764831845)
    parser.add_argument("--regenerate", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_digest(files: dict[str, dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(files[name]["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def scale_bits(scale: float) -> str:
    return struct.pack("<f", np.float32(scale)).hex()


def expected_contract(args: argparse.Namespace) -> dict[str, object]:
    hkv = args.heads if args.heads_kv is None else args.heads_kv
    if hkv == args.heads:
        return {
            "schema": SCHEMA,
            "formula_version": FORMULA_VERSION,
            "shape": {
                "batch": args.batch,
                "heads": args.heads,
                "seqlen_q": args.seqlen,
                "seqlen_k": args.seqlen,
                "head_dim_qk": args.dim,
                "head_dim_v": args.dim,
            },
            "causal": args.causal,
            "layout": "BHSD-contiguous",
            "input_dtype": "float16",
            "output_dtype": "float32",
            "softmax_scale_f32_bits": scale_bits(args.softmax_scale),
        }
    return {
        "schema": GQA_SCHEMA,
        "formula_version": GQA_FORMULA_VERSION,
        "shape": {
            "batch": args.batch,
            "heads_q": args.heads,
            "heads_kv": hkv,
            "seqlen_q": args.seqlen,
            "seqlen_k": args.seqlen,
            "head_dim_qk": args.dim,
            "head_dim_v": args.dim,
        },
        "causal": args.causal,
        "layout": "BHSD-contiguous",
        "input_dtype": "float16",
        "output_dtype": "float32",
        "softmax_scale_f32_bits": scale_bits(args.softmax_scale),
    }


def cache_key(args: argparse.Namespace) -> str:
    hkv = args.heads if args.heads_kv is None else args.heads_kv
    if hkv != args.heads:
        return (
            f"v2_b{args.batch}_hq{args.heads}_hkv{hkv}_s{args.seqlen}"
            f"_d{args.dim}_c{args.causal}_scale{scale_bits(args.softmax_scale)}"
        )
    return (
        f"v1_b{args.batch}_h{args.heads}_s{args.seqlen}_d{args.dim}"
        f"_c{args.causal}_scale{scale_bits(args.softmax_scale)}"
    )


def expected_file_specs(args: argparse.Namespace) -> dict[str, tuple[str, list[int]]]:
    b, hq, s, d = args.batch, args.heads, args.seqlen, args.dim
    hkv = hq if args.heads_kv is None else args.heads_kv
    q_tensor = [b, hq, s, d]
    kv_tensor = [b, hkv, s, d]
    rows = [b, hq, s]
    return {
        "q.f16": ("<f2", q_tensor),
        "k.f16": ("<f2", kv_tensor),
        "v.f16": ("<f2", kv_tensor),
        "o.f16": ("<f2", q_tensor),
        "dout.f16": ("<f2", q_tensor),
        "scores_max.f32": ("<f4", rows),
        "scores_sum.f32": ("<f4", rows),
        "delta.f32": ("<f4", rows),
        "packed_sidecar.f32": ("<f4", [b, hq, s, 3]),
        "dq.f32": ("<f4", q_tensor),
        "dk.f32": ("<f4", kv_tensor),
        "dv.f32": ("<f4", kv_tensor),
    }


def deterministic_tensor(
    count: int, mul: int, mod: int, value_scale: float
) -> np.ndarray:
    index = np.arange(count, dtype=np.int64)
    values = ((index * mul + 7) % mod) - (mod // 2)
    return (values.astype(np.float32) * np.float32(value_scale)).astype(np.float16)


def cpu_backward(args: argparse.Namespace) -> dict[str, np.ndarray]:
    b, hq, s, d = args.batch, args.heads, args.seqlen, args.dim
    hkv = hq if args.heads_kv is None else args.heads_kv
    q_count = b * hq * s * d
    kv_count = b * hkv * s * d
    q = deterministic_tensor(q_count, 3, 29, 0.009).reshape(b, hq, s, d)
    k = deterministic_tensor(kv_count, 5, 31, 0.008).reshape(b, hkv, s, d)
    v = deterministic_tensor(kv_count, 7, 37, 0.007).reshape(b, hkv, s, d)
    dout = deterministic_tensor(q_count, 11, 41, 0.006).reshape(b, hq, s, d)

    out = np.empty((b, hq, s, d), dtype=np.float16)
    scores_max = np.empty((b, hq, s), dtype=np.float32)
    scores_sum = np.empty((b, hq, s), dtype=np.float32)
    delta = np.empty((b, hq, s), dtype=np.float32)
    dq = np.empty((b, hq, s, d), dtype=np.float32)
    dk = np.zeros((b, hkv, s, d), dtype=np.float32)
    dv = np.zeros((b, hkv, s, d), dtype=np.float32)

    scale = np.float32(args.softmax_scale)
    scale_log2 = np.float32(scale * LOG2_E)
    causal_mask = np.triu(np.ones((s, s), dtype=bool), k=1)
    for batch_index in range(b):
        for head_index in range(hq):
            kv_head = head_index // (hq // hkv)
            qf = q[batch_index, head_index].astype(np.float32)
            kf = k[batch_index, kv_head].astype(np.float32)
            vf = v[batch_index, kv_head].astype(np.float32)
            dof = dout[batch_index, head_index].astype(np.float32)

            score = np.matmul(qf, kf.T).astype(np.float32)
            if args.causal:
                score = score.copy()
                score[causal_mask] = -np.inf
            row_max = np.max(score, axis=1).astype(np.float32)
            p_num = np.exp2((score - row_max[:, None]) * scale_log2).astype(
                np.float32
            )
            if args.causal:
                p_num[causal_mask] = np.float32(0.0)
            row_sum = np.sum(p_num, axis=1, dtype=np.float64).astype(np.float32)
            prob = (p_num / row_sum[:, None]).astype(np.float32)

            out_f16 = np.matmul(prob, vf).astype(np.float16)
            delta_row = np.sum(
                dof * out_f16.astype(np.float32), axis=1, dtype=np.float32
            ).astype(np.float32)
            dp = np.matmul(dof, vf.T).astype(np.float32)
            ds = (prob * (dp - delta_row[:, None]) * scale).astype(np.float32)

            out[batch_index, head_index] = out_f16
            scores_max[batch_index, head_index] = row_max
            scores_sum[batch_index, head_index] = row_sum
            delta[batch_index, head_index] = delta_row
            dq[batch_index, head_index] = np.matmul(ds, kf).astype(np.float32)
            dk[batch_index, kv_head] += np.matmul(ds.T, qf).astype(np.float32)
            dv[batch_index, kv_head] += np.matmul(prob.T, dof).astype(np.float32)

    packed_sidecar = np.empty((b, hq, s, 3), dtype=np.float32)
    packed_sidecar[..., 0] = scores_max * scale_log2
    packed_sidecar[..., 1] = np.where(
        scores_sum != 0.0, np.float32(1.0) / scores_sum, np.float32(0.0)
    )
    packed_sidecar[..., 2] = delta
    return {
        "q.f16": q,
        "k.f16": k,
        "v.f16": v,
        "o.f16": out,
        "dout.f16": dout,
        "scores_max.f32": scores_max,
        "scores_sum.f32": scores_sum,
        "delta.f32": delta,
        "packed_sidecar.f32": packed_sidecar,
        "dq.f32": dq,
        "dk.f32": dk,
        "dv.f32": dv,
    }


def write_cache(cache_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    root = cache_dir.parent
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{cache_dir.name}.tmp-", dir=root))
    try:
        arrays = cpu_backward(args)
        files: dict[str, dict[str, object]] = {}
        for name, array in arrays.items():
            dtype = np.dtype("<f2") if name.endswith(".f16") else np.dtype("<f4")
            payload = np.ascontiguousarray(array, dtype=dtype)
            path = temp_dir / name
            payload.tofile(path)
            files[name] = {
                "dtype": dtype.str,
                "shape": list(payload.shape),
                "nbytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        manifest = expected_contract(args)
        manifest["files"] = files
        manifest["payload_sha256"] = payload_digest(files)
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temp_dir, cache_dir)
        return manifest
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def validate_cache(cache_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = expected_contract(args)
    for key, expected in contract.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"manifest mismatch for {key}: expected={expected!r} "
                f"actual={manifest.get(key)!r}"
            )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("manifest has no files table")
    expected_specs = expected_file_specs(args)
    if set(files) != set(expected_specs):
        raise RuntimeError(
            f"cached payload set mismatch: expected={sorted(expected_specs)} "
            f"actual={sorted(files)}"
        )
    for name, metadata in files.items():
        path = cache_dir / name
        dtype, shape = expected_specs[name]
        expected_nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if metadata.get("dtype") != dtype or metadata.get("shape") != shape:
            raise RuntimeError(f"cached payload contract mismatch: {path}")
        if metadata.get("nbytes") != expected_nbytes:
            raise RuntimeError(f"cached payload byte contract mismatch: {path}")
        if not path.is_file():
            raise RuntimeError(f"missing cached payload: {path}")
        if path.stat().st_size != metadata.get("nbytes"):
            raise RuntimeError(f"cached payload size mismatch: {path}")
        if sha256_file(path) != metadata.get("sha256"):
            raise RuntimeError(f"cached payload hash mismatch: {path}")
    actual_payload = payload_digest(files)
    if actual_payload != manifest.get("payload_sha256"):
        raise RuntimeError("payload digest mismatch")
    return manifest


def main() -> int:
    args = parse_args()
    hkv = args.heads if args.heads_kv is None else args.heads_kv
    if min(args.batch, args.heads, hkv, args.seqlen, args.dim) <= 0:
        raise SystemExit("all shape dimensions must be positive")
    if args.heads % hkv != 0:
        raise SystemExit("heads must be divisible by heads-kv")
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache_dir = root / cache_key(args)

    if args.regenerate and cache_dir.exists():
        shutil.rmtree(cache_dir)
    if cache_dir.exists():
        manifest = validate_cache(cache_dir, args)
        status = "HIT"
    else:
        manifest = write_cache(cache_dir, args)
        manifest = validate_cache(cache_dir, args)
        status = "MISS"

    print(f"golden_cache_status={status}")
    print(f"golden_cache_path={cache_dir}")
    print(f"golden_payload_sha256={manifest['payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
