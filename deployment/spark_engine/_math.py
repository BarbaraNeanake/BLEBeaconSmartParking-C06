"""
Pure Python math operations using NumPy
Provides convolution, activation functions, batch normalization, and pooling
"""

import numpy as np
from typing import Tuple, Optional


def _im2col(x: np.ndarray, KH: int, KW: int, stride: int = 1, 
            pad: int = 0, dilation: int = 1) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Convert image to column matrix for efficient convolution
    
    Args:
        x: Input tensor (N, C, H, W)
        KH, KW: Kernel height and width
        stride: Stride size
        pad: Padding size
        dilation: Dilation rate
    
    Returns:
        Tuple of (col_matrix, output_shape)
    """
    N, C, H, W = x.shape
    
    # Add padding
    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    
    H_padded = x.shape[2]
    W_padded = x.shape[3]
    
    # Calculate output dimensions
    H_out = (H_padded - dilation * (KH - 1) - 1) // stride + 1
    W_out = (W_padded - dilation * (KW - 1) - 1) // stride + 1
    
    # Pre-allocate column matrix (N*H_out*W_out, C*KH*KW)
    col = np.zeros((N * H_out * W_out, C * KH * KW), dtype=x.dtype)
    
    idx = 0
    for n in range(N):
        for h_out in range(H_out):
            for w_out in range(W_out):
                h_start = h_out * stride
                w_start = w_out * stride
                
                # Extract patch and flatten
                patch = x[n, :, h_start:h_start + KH, w_start:w_start + KW]
                col[idx, :] = patch.reshape(-1)
                idx += 1
    
    return col, (H_out, W_out)


def _col2im(col: np.ndarray, output_shape: Tuple[int, int, int, int],
            KH: int, KW: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    N, C, H, W = output_shape
    H_padded = H + 2 * pad
    W_padded = W + 2 * pad
    
    img = np.zeros((N, C, H_padded, W_padded), dtype=col.dtype)
    
    H_out = (H_padded - KH) // stride + 1
    W_out = (W_padded - KW) // stride + 1
    
    idx = 0
    for n in range(N):
        for h_out in range(H_out):
            for w_out in range(W_out):
                h_start = h_out * stride
                w_start = w_out * stride
                
                patch = col[idx, :].reshape(C, KH, KW)
                img[n, :, h_start:h_start + KH, w_start:w_start + KW] += patch
                idx += 1
    
    # Remove padding
    if pad > 0:
        img = img[:, :, pad:-pad, pad:-pad]
    
    return img


def conv2d(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None,
           stride: int = 1, pad: int = 0, dilation: int = 1, groups: int = 1) -> np.ndarray:
    """
    2D convolution operation using im2col approach (efficient matrix multiplication)
    
    Args:
        x: Input tensor (N, C_in, H, W)
        weight: Conv weights (C_out, C_in/groups, KH, KW)
        bias: Optional bias (C_out,)
        stride: Stride size
        pad: Padding size
        dilation: Dilation rate
        groups: Number of groups for grouped convolution (not yet supported)
    
    Returns:
        Output tensor (N, C_out, H_out, W_out)
    """
    N, C_in, H, W = x.shape
    C_out, C_group, KH, KW = weight.shape
    
    # Convert to column format (more efficient than nested loops)
    col, (H_out, W_out) = _im2col(x, KH, KW, stride, pad, dilation)
    
    # Reshape weights to (C_out, C_in*KH*KW) for matrix multiplication
    weight_col = weight.reshape(C_out, -1)  # (C_out, C_in*KH*KW)
    
    # Perform convolution via matrix multiplication
    # col: (N*H_out*W_out, C_in*KH*KW)
    # weight_col.T: (C_in*KH*KW, C_out)
    # result: (N*H_out*W_out, C_out)
    out = np.dot(col, weight_col.T)  # (N*H_out*W_out, C_out)
    
    # Add bias
    if bias is not None:
        out = out + bias[np.newaxis, :]
    
    # Reshape to output tensor (N, C_out, H_out, W_out)
    out = out.reshape(N, H_out, W_out, C_out)
    out = np.transpose(out, (0, 3, 1, 2))  # (N, C_out, H_out, W_out)
    
    return out.astype(x.dtype)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation"""
    return np.maximum(x, 0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def batch_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               mean: np.ndarray, var: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Batch normalization
    
    Args:
        x: Input tensor (N, C, H, W)
        weight: Scale parameter (C,)
        bias: Shift parameter (C,)
        mean: Running mean (C,)
        var: Running variance (C,)
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor
    """
    # Normalize
    x_norm = (x - mean[np.newaxis, :, np.newaxis, np.newaxis]) / \
             np.sqrt(var[np.newaxis, :, np.newaxis, np.newaxis] + eps)
    
    # Scale and shift
    out = weight[np.newaxis, :, np.newaxis, np.newaxis] * x_norm + \
          bias[np.newaxis, :, np.newaxis, np.newaxis]
    
    return out.astype(x.dtype)


def max_pool2d(x: np.ndarray, kernel_size: int, stride: int = None, 
               padding: int = 0) -> np.ndarray:
    """
    Max pooling 2D
    
    Args:
        x: Input tensor (N, C, H, W)
        kernel_size: Size of pooling window
        stride: Stride (defaults to kernel_size if None)
        padding: Padding size
    
    Returns:
        Pooled tensor
    """
    if stride is None:
        stride = kernel_size
    
    N, C, H, W = x.shape
    
    # Add padding
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                   mode='constant', constant_values=-np.inf)
    
    H_padded = x.shape[2]
    W_padded = x.shape[3]
    
    # Calculate output dimensions
    H_out = (H_padded - kernel_size) // stride + 1
    W_out = (W_padded - kernel_size) // stride + 1
    
    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
    
    for h in range(H_out):
        for w in range(W_out):
            h_start = h * stride
            w_start = w * stride
            pool = x[:, :, h_start:h_start + kernel_size, w_start:w_start + kernel_size]
            out[:, :, h, w] = np.max(pool.reshape(N, C, -1), axis=2)
    
    return out.astype(x.dtype)


def adaptive_avg_pool2d(x: np.ndarray, output_size: int = 1) -> np.ndarray:
    """
    Adaptive average pooling
    
    Args:
        x: Input tensor (N, C, H, W)
        output_size: Output spatial size
    
    Returns:
        Pooled tensor (N, C, output_size, output_size)
    """
    N, C, H, W = x.shape
    out = np.zeros((N, C, output_size, output_size), dtype=x.dtype)
    
    stride_h = H // output_size
    stride_w = W // output_size
    
    for h in range(output_size):
        for w in range(output_size):
            h_start = h * stride_h
            w_start = w * stride_w
            h_end = h_start + stride_h
            w_end = w_start + stride_w
            
            pool = x[:, :, h_start:h_end, w_start:w_end]
            out[:, :, h, w] = np.mean(pool.reshape(N, C, -1), axis=2)
    
    return out.astype(x.dtype)
