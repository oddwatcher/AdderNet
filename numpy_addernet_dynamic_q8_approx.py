import torch
import numpy as np
from numpy.lib.stride_tricks import as_strided
from adder_approx import approx_sum_B, approx_sum_C


def joint_quantize(tensor1, tensor2, qmin=-(2**31), qmax=2**31 - 1):
    # Should be done at train-quantize process however mother f*king torch does not support quantization of custom modules
    max_val = np.max(np.abs(np.concatenate([tensor1.flatten(), tensor2.flatten()])))
    scale = max_val / qmax if max_val != 0 else 1.0
    q_tensor1 = np.round(tensor1 / scale).astype(np.int32)
    q_tensor2 = np.round(tensor2 / scale).astype(np.int32)
    q_tensor1 = np.clip(q_tensor1, qmin, qmax)
    q_tensor2 = np.clip(q_tensor2, qmin, qmax)
    return q_tensor1, q_tensor2, scale


def im2col_indices(x, kh, kw, padding=0, stride=1):
    n_x, d, h, w = x.shape

    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1

    if padding > 0:
        x = np.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0,
        )
    else:
        x = x.copy()

    shape = (n_x, d, kh, kw, h_out, w_out)
    strides = (
        x.strides[0],
        x.strides[1],
        x.strides[2],
        x.strides[3],
        stride * x.strides[2],
        stride * x.strides[3],
    )
    x_strided = as_strided(x, shape=shape, strides=strides, writeable=False)

    return x_strided.reshape(d * kh * kw, n_x * h_out * w_out)


def forward_conv2d(X, W, b=None, stride=1, padding=0):
    n_x, d_x, h_x, w_x = X.shape
    n_filters, d_filter, h_filter, w_filter = W.shape
    assert d_x == d_filter, "input channels must met filter channels"

    cols = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)

    W_col = W.reshape(n_filters, -1)  # (n_filters, d * kh * kw)

    output = W_col @ cols  # (n_filters, n_x * h_out * w_out)

    h_out = (h_x + 2 * padding - h_filter) // stride + 1
    w_out = (w_x + 2 * padding - w_filter) // stride + 1
    h_out, w_out = int(h_out), int(w_out)

    output = output.reshape(n_filters, n_x, h_out, w_out).transpose(1, 0, 2, 3)

    if b is not None:
        output += b.reshape(1, -1, 1, 1)

    return output


def forward_adder2d(X, W, stride=1, padding=0, bias=None):
    n_x, d_x, h_x, w_x = X.shape
    n_filters, d_filter, h_filter, w_filter = W.shape
    assert d_x == d_filter
    h_out = (h_x + 2 * padding - h_filter) // stride + 1
    w_out = (w_x + 2 * padding - w_filter) // stride + 1
    h_out, w_out = int(h_out), int(w_out)

    cols = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    output = -np.abs(W_col[:, :, np.newaxis] - cols[np.newaxis, :, :]).sum(axis=1)

    output = output.reshape(n_filters, n_x, h_out, w_out).transpose(1, 0, 2, 3)

    if bias is not None:
        output += bias.reshape(1, -1, 1, 1)

    return output


def forward_adder2d_approx(X, W, stride=1, padding=0, bias=None):
    global approx_bits
    n_x, d_x, h_x, w_x = X.shape
    n_filters, d_filter, h_filter, w_filter = W.shape
    assert d_x == d_filter
    h_out = (h_x + 2 * padding - h_filter) // stride + 1
    w_out = (w_x + 2 * padding - w_filter) // stride + 1
    h_out, w_out = int(h_out), int(w_out)

    cols = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    W_q, cols_q, scale = joint_quantize(W_col, cols)

    output = -np.abs(
        approx_sum_B(W_q[:, :, np.newaxis], -cols_q[np.newaxis, :, :], approx_bits)
    )
    output = np.sum(output, axis=1) * scale
    output = output.reshape(n_filters, n_x, h_out, w_out).transpose(1, 0, 2, 3)

    if bias is not None:
        output += bias.reshape(1, -1, 1, 1)

    return output


def forward_adder2d_quantize(X, W, stride=1, padding=0, bias=None):
    n_x, d_x, h_x, w_x = X.shape
    n_filters, d_filter, h_filter, w_filter = W.shape
    assert d_x == d_filter
    h_out = (h_x + 2 * padding - h_filter) // stride + 1
    w_out = (w_x + 2 * padding - w_filter) // stride + 1
    h_out, w_out = int(h_out), int(w_out)

    cols = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    W_q, cols_q, scale = joint_quantize(W_col, cols)

    output = -np.abs(W_q[:, :, np.newaxis] - cols_q[np.newaxis, :, :])
    output = np.sum(output, axis=1) * scale
    output = output.reshape(n_filters, n_x, h_out, w_out).transpose(1, 0, 2, 3)

    if bias is not None:
        output += bias.reshape(1, -1, 1, 1)

    return output


def forward_batchnorm2d(X, weight, bias, running_mean, running_var, eps=1e-5):
    mean = running_mean.reshape(1, -1, 1, 1)
    var = running_var.reshape(1, -1, 1, 1)
    weight = weight.reshape(1, -1, 1, 1)
    bias = bias.reshape(1, -1, 1, 1)
    return weight * (X - mean) / np.sqrt(var + eps) + bias


def relu(x):
    return np.maximum(0, x)


def forward_avgpool2d(X, kernel_size, stride=None):
    if stride is None:
        stride = kernel_size

    N, C, H, W = X.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    output = np.zeros((N, C, H_out, W_out))

    for h in range(H_out):
        for w in range(W_out):
            h_start = h * stride
            h_end = h_start + kernel_size
            w_start = w * stride
            w_end = w_start + kernel_size
            output[:, :, h, w] = X[:, :, h_start:h_end, w_start:w_end].mean(axis=(2, 3))

    return output


class ResNetNumpy:
    def __init__(self, params):
        self.params = params

    def forward(self, X):

        X = forward_conv2d(X, self.params["conv1"]["weight"], padding=1)
        X = forward_batchnorm2d(X, **self.params["bn1"])
        X = relu(X)

        for i in range(3):
            X = self._forward_basic_block(X, self.params["layer1"][f"block{i}"])

        for i in range(3):
            X = self._forward_basic_block(X, self.params["layer2"][f"block{i}"])

        for i in range(3):
            X = self._forward_basic_block(X, self.params["layer3"][f"block{i}"])

        X = forward_avgpool2d(X, kernel_size=8)

        X = forward_conv2d(X, self.params["fc"]["weight"])

        X = forward_batchnorm2d(X, **self.params["bn2"])

        return X.reshape(X.shape[0], -1)

    def _forward_basic_block(self, X, block_params):
        residual = X

        X = forward_adder2d_approx(
            X,
            block_params["conv1"]["weight"],
            stride=block_params["conv1"]["stride"],
            padding=block_params["conv1"]["padding"],
        )

        X = forward_batchnorm2d(X, **block_params["bn1"])
        X = relu(X)
        X = forward_adder2d_approx(
            X,
            block_params["conv2"]["weight"],
            stride=block_params["conv2"]["stride"],
            padding=block_params["conv2"]["padding"],
        )
        X = forward_batchnorm2d(X, **block_params["bn2"])

        if block_params.get("downsample", None):
            residual = forward_adder2d_approx(
                residual,
                block_params["downsample"]["adder"]["weight"],
                stride=block_params["downsample"]["stride"],
                padding=0,
            )
            residual = forward_batchnorm2d(residual, **block_params["downsample"]["bn"])

        X += residual
        X = relu(X)
        return X


def load_params(state_dict_torch):
    params = {}

    params["conv1"] = {
        "weight": state_dict_torch["conv1.weight"],
    }

    params["bn1"] = {
        "weight": state_dict_torch["bn1.weight"],
        "bias": state_dict_torch["bn1.bias"],
        "running_mean": state_dict_torch["bn1.running_mean"],
        "running_var": state_dict_torch["bn1.running_var"],
    }

    for layer in ["layer1", "layer2", "layer3"]:
        params[layer] = {}
        for i in range(3):
            block_prefix = f"{layer}.{i}"
            block = {}

            block["conv1"] = {
                "weight": state_dict_torch[f"{block_prefix}.conv1.adder"],
                "bias": state_dict_torch.get(f"{block_prefix}.conv1.b", None),
                "stride": 2 if layer != "layer1" and i == 0 else 1,
                "padding": 1,
            }

            block["bn1"] = {
                "weight": state_dict_torch[f"{block_prefix}.bn1.weight"],
                "bias": state_dict_torch[f"{block_prefix}.bn1.bias"],
                "running_mean": state_dict_torch[f"{block_prefix}.bn1.running_mean"],
                "running_var": state_dict_torch[f"{block_prefix}.bn1.running_var"],
            }

            block["conv2"] = {
                "weight": state_dict_torch[f"{block_prefix}.conv2.adder"],
                "bias": state_dict_torch.get(f"{block_prefix}.conv2.b", None),
                "stride": 1,
                "padding": 1,
            }

            block["bn2"] = {
                "weight": state_dict_torch[f"{block_prefix}.bn2.weight"],
                "bias": state_dict_torch[f"{block_prefix}.bn2.bias"],
                "running_mean": state_dict_torch[f"{block_prefix}.bn2.running_mean"],
                "running_var": state_dict_torch[f"{block_prefix}.bn2.running_var"],
            }

            downsample_prefix = f"{block_prefix}.downsample"
            if f"{downsample_prefix}.0.adder" in state_dict_torch:
                block["downsample"] = {
                    "adder": {
                        "weight": state_dict_torch[f"{downsample_prefix}.0.adder"],
                        "bias": state_dict_torch.get(f"{downsample_prefix}.0.b", None),
                    },
                    "bn": {
                        "weight": state_dict_torch[f"{downsample_prefix}.1.weight"],
                        "bias": state_dict_torch[f"{downsample_prefix}.1.bias"],
                        "running_mean": state_dict_torch[
                            f"{downsample_prefix}.1.running_mean"
                        ],
                        "running_var": state_dict_torch[
                            f"{downsample_prefix}.1.running_var"
                        ],
                    },
                    "stride": 2,
                }
            else:
                block["downsample"] = None

            params[layer][f"block{i}"] = block

    params["fc"] = {
        "weight": state_dict_torch["fc.weight"],
    }

    params["bn2"] = {
        "weight": state_dict_torch["bn2.weight"],
        "bias": state_dict_torch["bn2.bias"],
        "running_mean": state_dict_torch["bn2.running_mean"],
        "running_var": state_dict_torch["bn2.running_var"],
    }

    return params


if __name__ == "__main__":
    from tqdm import tqdm
    from torchvision.datasets import CIFAR10
    from torchvision import transforms

    global approx_bits
    approx_bits = 0
    state_dict = torch.load("trained/addernet_CIFAR10_best.pt",map_location='cuda:0')

    state_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}
    params = load_params(state_dict)
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    data_test = CIFAR10("./cache/data/", train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=True)
    resnet_numpy = ResNetNumpy(params)
    log = "TypeB_adder-resnet20_CIFAR10_q32.txt"

    for i in range(1, 33):
        approx_bits = i
        correct = 0
        total = 0
        with torch.no_grad():
            for num, pair in enumerate(tqdm(test_loader, total=len(test_loader))):
                images, labels = pair
                if num > 2000:
                    break
                images_np = images.numpy()

                outputs = resnet_numpy.forward(images_np)

                predicted = np.argmax(outputs, axis=1)

                total += labels.size(0)
                correct += (predicted == labels.numpy()).sum().item()
                print(
                    f"Test Accuracy: {100 * correct / total:.2f}% approx_bits:{approx_bits}"
                )

        print(f"Test Accuracy: {100 * correct / total:.2f}% approx_bits:{approx_bits}")
        with open(log, "a") as logout:
            logout.write(
                f"Test Accuracy: {100 * correct / total:.2f}% approx_bits:{approx_bits}\n"
            )
