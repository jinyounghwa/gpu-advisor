"""
Model Quantization (FP32 -> INT8)
Optimize for Mac M4 minimal memory
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Optional, Dict, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantizationConfig:
    """Quantization configuration"""

    def __init__(
        self,
        backend: str = "qnnpack",
        dtype: str = "qint8",
        observer_type: str = "average",
    ):
        self.backend = backend
        self.dtype = dtype
        self.observer_type = observer_type


class ModelQuantizer:
    """
    Model Quantization Wrapper
    """

    def __init__(self, config: QuantizationConfig = None):
        self.config = config or QuantizationConfig()

    def quantize_static(
        self, model: nn.Module, calibration_loader, save_path: Optional[str] = None
    ) -> nn.Module:
        """
        Static quantization (requires calibration data)

        Args:
            model: PyTorch model to quantize
            calibration_loader: DataLoader for calibration
            save_path: path to save quantized model

        Returns:
            quantized model
        """
        logger.info("Starting static quantization...")

        model.eval()
        model.qconfig = quant.get_default_qconfig(self.config.backend)

        model_prepared = quant.prepare(model, inplace=True)

        logger.info("Calibrating model...")
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_loader):
                model_prepared(data)
                if batch_idx >= 100:
                    break

        model_quantized = quant.convert(model_prepared, inplace=True)

        if save_path:
            torch.jit.save(torch.jit.script(model_quantized), save_path)
            logger.info(f"Quantized model saved to {save_path}")

        return model_quantized

    def quantize_dynamic(
        self, model: nn.Module, save_path: Optional[str] = None
    ) -> nn.Module:
        """
        Dynamic quantization (no calibration needed)

        Args:
            model: PyTorch model to quantize
            save_path: path to save quantized model

        Returns:
            quantized model
        """
        logger.info("Starting dynamic quantization...")

        model.eval()

        model_quantized = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv1d, nn.Conv2d}, dtype=torch.qint8
        )

        if save_path:
            torch.jit.save(torch.jit.script(model_quantized), save_path)
            logger.info(f"Quantized model saved to {save_path}")

        return model_quantized

    def get_model_size(self, model: nn.Module) -> Dict[str, int]:
        """
        Get model size in bytes

        Args:
            model: PyTorch model

        Returns:
            dict with size information
        """
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.numel() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()

        total_size = param_size + buffer_size

        return {
            "param_size_mb": param_size / (1024**2),
            "buffer_size_mb": buffer_size / (1024**2),
            "total_size_mb": total_size / (1024**2),
            "total_size_bytes": total_size,
        }

    def compare_models(
        self,
        model_fp32: nn.Module,
        model_quantized: nn.Module,
        sample_input: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Compare FP32 and quantized models

        Args:
            model_fp32: original FP32 model
            model_quantized: quantized model
            sample_input: input tensor for testing

        Returns:
            dict with comparison metrics
        """
        model_fp32.eval()
        model_quantized.eval()

        with torch.no_grad():
            output_fp32 = model_fp32(sample_input)
            output_quantized = model_quantized(sample_input)

        size_fp32 = self.get_model_size(model_fp32)
        size_quantized = self.get_model_size(model_quantized)

        compression_ratio = (
            size_fp32["total_size_bytes"] / size_quantized["total_size_bytes"]
        )

        error = torch.abs(output_fp32 - output_quantized).mean().item()

        return {
            "compression_ratio": compression_ratio,
            "size_fp32_mb": size_fp32["total_size_mb"],
            "size_quantized_mb": size_quantized["total_size_mb"],
            "mean_error": error,
        }


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    from models.transformer_model import PolicyValueNetwork

    model = PolicyValueNetwork(
        input_dim=11, d_model=256, num_heads=8, num_layers=4, d_ff=512, num_actions=5
    ).to(device)

    model.eval()

    sample_input = torch.randn(1, 10, 11).to(device)

    quantizer = ModelQuantizer()

    model_quantized = quantizer.quantize_dynamic(model)

    logger.info("Original model size:")
    original_size = quantizer.get_model_size(model)
    logger.info(f"  Total: {original_size['total_size_mb']:.2f} MB")
    logger.info(f"  Params: {original_size['param_size_mb']:.2f} MB")
    logger.info(f"  Buffers: {original_size['buffer_size_mb']:.2f} MB")

    logger.info("\nQuantized model size:")
    quantized_size = quantizer.get_model_size(model_quantized)
    logger.info(f"  Total: {quantized_size['total_size_mb']:.2f} MB")
    logger.info(f"  Params: {quantized_size['param_size_mb']:.2f} MB")
    logger.info(f"  Buffers: {quantized_size['buffer_size_mb']:.2f} MB")

    comparison = quantizer.compare_models(model, model_quantized, sample_input)
    logger.info(f"\nCompression ratio: {comparison['compression_ratio']:.2f}x")
    logger.info(f"Mean error: {comparison['mean_error']:.6f}")
