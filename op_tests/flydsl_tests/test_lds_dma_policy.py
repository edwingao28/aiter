# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "aiter" / "ops" / "flydsl" / "kernels" / "lds_dma_policy.py"
MFMA_POLICY_PATH = REPO_ROOT / "aiter" / "ops" / "flydsl" / "kernels" / "mfma_policy.py"
MIXED_MOE_PATH = (
    REPO_ROOT / "aiter" / "ops" / "flydsl" / "kernels" / "mixed_moe_gemm_2stage.py"
)
PRESHUFFLE_PIPELINE_PATH = (
    REPO_ROOT / "aiter" / "ops" / "flydsl" / "kernels" / "mfma_preshuffle_pipeline.py"
)


def load_policy_module(path: Path, module_name: str):
    if not path.is_file():
        raise AssertionError(f"missing policy module: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load policy module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRawPtrBufferLoadLdsPolicy(unittest.TestCase):
    def test_gfx94_uses_legal_dword_width(self) -> None:
        select = load_policy_module(
            POLICY_PATH, "lds_dma_policy"
        ).select_raw_ptr_buffer_load_lds_bytes

        for arch in ("gfx940", "gfx941", "gfx942", "gfx942:sramecc+:xnack-"):
            with self.subTest(arch=arch):
                self.assertEqual(select(arch, 64), 4)

    def test_existing_architectures_keep_dwordx4_when_legal(self) -> None:
        select = load_policy_module(
            POLICY_PATH, "lds_dma_policy"
        ).select_raw_ptr_buffer_load_lds_bytes

        for arch in ("gfx950", "gfx1250", "unknown"):
            with self.subTest(arch=arch):
                self.assertEqual(select(arch, 64), 16)

    def test_non_dword_copy_size_is_rejected(self) -> None:
        select = load_policy_module(
            POLICY_PATH, "lds_dma_policy"
        ).select_raw_ptr_buffer_load_lds_bytes

        for size in (0, 2, 6):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "divisible by 4"):
                    select("gfx942", size)

    def test_non_multiple_of_16_falls_back_to_dword(self) -> None:
        select = load_policy_module(
            POLICY_PATH, "lds_dma_policy"
        ).select_raw_ptr_buffer_load_lds_bytes

        self.assertEqual(select("gfx950", 12), 4)

    def test_both_a16w4_stages_use_the_architecture_policy(self) -> None:
        source = MIXED_MOE_PATH.read_text()
        tree = ast.parse(source)
        functions = {
            node.name: ast.get_source_segment(source, node)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        for name in (
            "compile_mixed_moe_gemm1_a16w4",
            "compile_mixed_moe_gemm2_a16w4",
        ):
            with self.subTest(name=name):
                function_source = functions[name]
                function_tree = ast.parse(function_source)
                calls = [
                    node
                    for node in ast.walk(function_tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "select_raw_ptr_buffer_load_lds_bytes"
                ]
                self.assertTrue(
                    any(
                        [isinstance(arg, ast.Name) and arg.id for arg in call.args[:2]]
                        == ["gpu_arch", "bytes_per_thread_x"]
                        for call in calls
                    ),
                    f"{name} does not select DMA width from arch and copy size",
                )
                dma_assignments = [
                    node
                    for node in ast.walk(function_tree)
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "_dma_bytes"
                        for target in node.targets
                    )
                ]
                self.assertTrue(
                    any(
                        isinstance(node.value, ast.Name)
                        and node.value.id == "x_load_bytes"
                        for node in dma_assignments
                    ),
                    f"{name} does not use the selected width for the intrinsic",
                )
                self.assertFalse(
                    any(
                        isinstance(node.value, ast.Constant) and node.value.value == 16
                        for node in dma_assignments
                    ),
                    f"{name} still hard-codes a 16-byte DMA",
                )


class TestA16W4Bf16MfmaPolicy(unittest.TestCase):
    def test_gfx94_uses_legacy_k16_mfma(self) -> None:
        select = load_policy_module(
            MFMA_POLICY_PATH, "mfma_policy"
        ).select_a16w4_bf16_mfma_k

        for arch in ("gfx940", "gfx941", "gfx942", "gfx942:sramecc+:xnack-"):
            with self.subTest(arch=arch):
                self.assertEqual(select(arch), 16)

    def test_newer_architectures_keep_k32_mfma(self) -> None:
        select = load_policy_module(
            MFMA_POLICY_PATH, "mfma_policy"
        ).select_a16w4_bf16_mfma_k

        for arch in ("gfx950", "gfx1250", "unknown"):
            with self.subTest(arch=arch):
                self.assertEqual(select(arch), 32)

    def test_both_a16w4_stages_split_k32_work_on_gfx94(self) -> None:
        source = MIXED_MOE_PATH.read_text()
        tree = ast.parse(source)
        functions = {
            node.name: ast.get_source_segment(source, node)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        for name in (
            "compile_mixed_moe_gemm1_a16w4",
            "compile_mixed_moe_gemm2_a16w4",
        ):
            with self.subTest(name=name):
                function_source = functions[name]
                function_tree = ast.parse(function_source)
                policy_calls = [
                    node
                    for node in ast.walk(function_tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "select_a16w4_bf16_mfma_k"
                ]
                self.assertTrue(
                    any(
                        len(call.args) == 1
                        and isinstance(call.args[0], ast.Name)
                        and call.args[0].id == "gpu_arch"
                        for call in policy_calls
                    ),
                    f"{name} does not select the BF16 MFMA width from gpu_arch",
                )
                self.assertIn("mfma_f32_16x16x16bf16_1k", function_source)
                self.assertIn("def _mfma_k64(", function_source)

                mfma_k64 = next(
                    node
                    for node in ast.walk(function_tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "_mfma_k64"
                )
                k16_calls = [
                    node
                    for node in ast.walk(mfma_k64)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "mfma_f32_bf16_k16"
                ]
                self.assertEqual(
                    len(k16_calls),
                    2,
                    f"{name} must cover each K32 tile with two K16 MFMAs",
                )
                fp4_unpack_calls = [
                    node
                    for node in ast.walk(function_tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "unpack_b_mxfp4_bf16"
                ]
                self.assertTrue(
                    fp4_unpack_calls,
                    f"{name} does not contain an MXFP4 decode",
                )
                for call in fp4_unpack_calls:
                    use_hw_cvt = next(
                        (
                            keyword.value
                            for keyword in call.keywords
                            if keyword.arg == "use_hw_cvt"
                        ),
                        None,
                    )
                    self.assertIsNotNone(
                        use_hw_cvt,
                        f"{name} does not gate the gfx950-only FP4 conversion",
                    )
                    self.assertEqual(
                        ast.dump(use_hw_cvt),
                        ast.dump(ast.parse("bf16_mfma_k == 32", mode="eval").body),
                        f"{name} must use software FP4 decode with the gfx94 K16 path",
                    )

    def test_software_mxfp4_decode_preserves_k16_operand_order(self) -> None:
        source = PRESHUFFLE_PIPELINE_PATH.read_text()
        tree = ast.parse(source)
        function_source = next(
            ast.get_source_segment(source, node)
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_unpack_b_mxfp4_bf16_sw"
        )

        self.assertIn("n0 = packed32 & c_0f", function_source)
        self.assertIn(
            "n1 = arith.shrui(packed32, c4) & c_0f",
            function_source,
        )
        self.assertIn(
            "first = n0 | arith.shli(n1, c8)",
            function_source,
        )
        self.assertIn(
            "second = n4 | arith.shli(n5, c8)",
            function_source,
        )


if __name__ == "__main__":
    unittest.main()
