"""Export trained FacadeDrone policy to ONNX format.

Usage:
    python scripts/export_onnx.py --checkpoint logs/latest/nn/best.pth \
        --output models/facade_policy.onnx

Output:
    1. facade_policy.onnx — ONNX model file
    2. Validation: compares PyTorch vs ONNX outputs (max_diff < 1e-3)

To convert to TensorRT on Manifold 3 (Jetson Orin):
    trtexec --onnx=facade_policy.onnx --saveEngine=facade_policy_fp16.engine --fp16
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch


# Policy dimensions from PLAN.md
OBS_DIM = 23
ACT_DIM = 4


def build_policy() -> torch.nn.Module:
    """Build the MLP policy: [23] → 256 → 128 → 64 → [4]."""
    return torch.nn.Sequential(
        torch.nn.Linear(OBS_DIM, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 64),
        torch.nn.ELU(),
        torch.nn.Linear(64, ACT_DIM),
        torch.nn.Tanh(),
    )


def export_onnx(checkpoint_path: str, output_path: str):
    """Export trained policy checkpoint to ONNX."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" in checkpoint:
        model_state = checkpoint["model"]
    else:
        model_state = checkpoint

    # Build and load policy
    policy = build_policy()
    policy.load_state_dict(model_state, strict=False)
    policy.eval()

    # Count parameters
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {num_params:,} ({num_params/1000:.1f}K)")

    # Export to ONNX
    dummy_input = torch.randn(1, OBS_DIM)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        policy,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
    )
    print(f"ONNX model exported to: {output_path}")

    # Validate: compare PyTorch vs ONNX outputs
    _validate_onnx(policy, output_path)


def _validate_onnx(policy: torch.nn.Module, onnx_path: str):
    """Validate ONNX model against PyTorch model."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)

    # Test with 100 random inputs
    max_diff = 0.0
    for _ in range(100):
        test_input = torch.randn(1, OBS_DIM)

        # PyTorch output
        with torch.no_grad():
            torch_output = policy(test_input).numpy()

        # ONNX output
        onnx_output = session.run(None, {"observation": test_input.numpy()})[0]

        diff = np.abs(torch_output - onnx_output).max()
        max_diff = max(max_diff, diff)

    print(f"Validation: max absolute difference = {max_diff:.6f}")
    if max_diff < 1e-3:
        print("PASSED: PyTorch and ONNX outputs match within tolerance.")
    else:
        print(f"WARNING: Difference {max_diff:.6f} exceeds 1e-3 tolerance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FacadeDrone policy to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint.")
    parser.add_argument("--output", type=str, default="models/facade_policy.onnx", help="Output ONNX path.")
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.output)
