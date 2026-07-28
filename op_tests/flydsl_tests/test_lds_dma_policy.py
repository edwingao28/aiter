# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "aiter" / "ops" / "flydsl" / "kernels" / "lds_dma_policy.py"
MIXED_MOE_PATH = (
    REPO_ROOT / "aiter" / "ops" / "flydsl" / "kernels" / "mixed_moe_gemm_2stage.py"
)


def load_policy_module():
    if not POLICY_PATH.is_file():
        raise AssertionError(f"missing DMA policy module: {POLICY_PATH}")
    spec = importlib.util.spec_from_file_location("lds_dma_policy", POLICY_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load DMA policy module: {POLICY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRawPtrBufferLoadLdsPolicy(unittest.TestCase):
    def test_gfx94_uses_legal_dword_width(self) -> None:
        select = load_policy_module().select_raw_ptr_buffer_load_lds_bytes

        for arch in ("gfx940", "gfx941", "gfx942", "gfx942:sramecc+:xnack-"):
            with self.subTest(arch=arch):
                self.assertEqual(select(arch, 64), 4)

    def test_existing_architectures_keep_dwordx4_when_legal(self) -> None:
        select = load_policy_module().select_raw_ptr_buffer_load_lds_bytes

        for arch in ("gfx950", "gfx1250", "unknown"):
            with self.subTest(arch=arch):
                self.assertEqual(select(arch, 64), 16)

    def test_non_dword_copy_size_is_rejected(self) -> None:
        select = load_policy_module().select_raw_ptr_buffer_load_lds_bytes

        for size in (0, 2, 6):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "divisible by 4"):
                    select("gfx942", size)

    def test_non_multiple_of_16_falls_back_to_dword(self) -> None:
        select = load_policy_module().select_raw_ptr_buffer_load_lds_bytes

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


if __name__ == "__main__":
    unittest.main()
