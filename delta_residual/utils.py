import inspect
import os
import sys
import threading
from functools import partial
from typing import Any, Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from delta_residual.general_delta import AbstractDeltaModule
from loguru import logger
from torch.multiprocessing import Lock

# logger.remove()
# logger.add(sys.stderr, level=os.environ.get("LOGLEVEL", "DEBUG"))


def get_module_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def get_tuple_device(t: Tuple) -> torch.device:
    for item in t:
        if isinstance(item, torch.Tensor):
            return item.device
    return None


def ModuleDeviceAddOn(cls):
    cls.device = property(lambda self: get_module_device(self))
    return cls


# 这里是一个partial
# @nn.DataParallel
#  现在我是钩子函数，我会被来自不同device的东西tensor调用

# class AutoDeviceModuleForSelfHook(nn.Module):
#     """Some Information about ModuleForSelfHook"""

#     def __init__(self, self_delta_model: nn.Module, hook_without_self: Callable):
#         super().__init__()
#         self.self_delta_model = self_delta_model
#         self.hook_without_self = hook_without_self

#     def forward(self, *args, **kwargs):
#         return self.hook_without_self(self.self_delta_model, *args, **kwargs)
# device = get_tuple_device(args)
# if device is None:
#     return self.hook_without_self(self.self_delta_model, *args, **kwargs)
# return self.hook_without_self(self.self_delta_model.to(device), *args, **kwargs)


class ModuleForSelfHook(nn.Module):
    """Some Information about ModuleForSelfHook"""

    def __init__(self, self_delta_model: nn.Module, hook_without_self: Callable):
        super(ModuleForSelfHook, self).__init__()
        self.self_delta_model = self_delta_model
        self.hook_without_self = hook_without_self

    def forward(self, *args, **kwargs):
        return self.hook_without_self(self.self_delta_model, *args, **kwargs)


from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter_kwargs


@ModuleDeviceAddOn
class AutoToDevice(nn.Module):
    """Some Information about AutoToDevice"""

    def __init__(self, original_module: nn.Module):
        super(AutoToDevice, self).__init__()
        self.original_module = original_module
        # self.original_module = original_module.cuda() # 使用relicate保证正确
        self.lock = Lock()  # 保证同时这个Module只在一个device

    def forward(self, *args, **kwargs):
        device = get_tuple_device(args)
        if device is None:
            return self.original_module.forward(*args, **kwargs)
        with self.lock:
            backup = self.__class__.__getattr__

            def temp_getattr(self, name: str):
                if "_parameters" in self.__dict__:
                    _parameters = self.__dict__["_parameters"]
                    if name in _parameters:
                        # return _parameters[name].to(device).requires_grad_()
                        return _parameters[name]
                return backup(self, name)

            # self.original_module.__class__.__getattr__ = temp_getattr
            nn.Module.__getattr__ = temp_getattr

            result = self.original_module(*args, **kwargs)

            # self.original_module.__class__.__getattr__ = backup
            nn.Module.__getattr__ = backup

            # logger.info(
            #     f"AutoToDevice: input is from {device}, my device is {self.device}. "
            # )
            # self.to(device) # 似乎不行，计算图会中断？
            # result = self.original_module.forward(*args, **kwargs)

            # dim = 0
            # device_ids = [device]
            # # args, kwargs = scatter_kwargs(args, kwargs, device_ids, dim=dim)
            # replicas = replicate(self.original_module, device_ids)
            # outputs = parallel_apply(replicas, args, kwargs, device_ids)
            # result = gather(outputs, device, dim=dim)

        return result


class AutoDeviceModuleForSelfHook(nn.Module):
    """Some Information about AutoParallelModuleForSelfHook"""

    def __init__(
        self,
        self_delta_model: nn.Module,
        hook_without_self: Callable,
        auto_to_device_cls=AutoToDevice,  # 也可能是 nn.DataParallel
        # auto_to_device_cls=nn.DataParallel,  # 也可能是 nn.DataParallel
    ):
        super(AutoDeviceModuleForSelfHook, self).__init__()
        self.paralleled = auto_to_device_cls(
            ModuleForSelfHook(self_delta_model, hook_without_self)
        )

    def forward(self, *args, **kwargs):
        return self.paralleled(*args, **kwargs)


def get_param_norm(module: nn.Module):
    return torch.sqrt(sum([torch.norm(p) ** 2 for p in module.parameters()]))


def get_sorted_function_inputs_from_args(fun, *args, **kwargs) -> dict[str, Any]:
    args = list(args)
    # 将输入函数的参数正则化，变成按照函数调用顺序的、写出参数名称的字典调用
    sig = inspect.signature(fun)
    result = dict()  # python字典是有序的。
    # kwargs的顺序不一定和函数定义的顺序一致，所以要先把kwargs里的参数按照函数定义的顺序放到result里。
    length_args = len(args)
    for i, (name, param) in enumerate(sig.parameters.items()):
        kind = param.kind
        # TODO 暂时不支持 forward本身也有*args, **kwargs的情况， 如果是函数的中间有这两玩意更加恐怖。
        if (
            kind == inspect.Parameter.VAR_POSITIONAL
            or kind == inspect.Parameter.VAR_KEYWORD
        ):
            raise TypeError(f'"{kind}" is not supported in forward.')

        # 这个是因为 onnx.export 必须要以 positional arguments的形式传进去。
        if kind == inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(f'"{kind}" is not supported in forward.')
        if i < length_args:
            item = args.pop(0)
            result[name] = item
        else:
            # result[name] = ... # 先占一个位置。为了保证和Python行为一样
            if name in kwargs:
                result[name] = kwargs.pop(name)
            elif param.default is not inspect._empty:
                result[name] = param.default
            else:
                raise TypeError(f'missing a required argument "{name}"')
    # args 和 kwargs可能有剩余的参数，这就是错误
    if len(args) > 0:
        raise TypeError("forward() got too many positional arguments")
    if len(kwargs) > 0:
        raise TypeError("forward() got too many keyword arguments")
    return result


def auto_tuple_output_for_forward_hook(
    unwrapped_hook: Callable[[nn.Module, nn.Module, Tuple, Tuple], Tuple]
) -> Callable[[nn.Module, nn.Module, Tuple, Tuple | torch.Tensor], Tuple]:
    def wrapped_hook(
        self, module: nn.Module, inputs: tuple, outputs: tuple | torch.Tensor
    ) -> tuple:
        # logger.warning(self)
        not_tuple = False
        if not isinstance(outputs, tuple):
            not_tuple = True
            outputs = (outputs,)

        new_outputs = unwrapped_hook(self, module, inputs, outputs)

        if not_tuple:
            return new_outputs[0]
        else:
            return new_outputs

    return wrapped_hook


def set_requires_grad(model: nn.Module, requires_grad: bool = False) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad


# TODO 实际上我们的mechanism不一样。我们的delta和model是完全分离的，仅仅使用hook的方式进行了微弱的连接。
def SeeTrainableParametersAddOn(cls):
    cls.get_nb_trainable_parameters = get_nb_trainable_parameters
    cls.print_trainable_parameters = print_trainable_parameters
    return cls


# The following functions `get_nb_trainable_parameters` and `print_trainable_parameters` are borrowed from huggingface/peft.
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def get_nb_trainable_parameters(self: nn.Module) -> tuple[int, int]:
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """

    trainable_params = 0
    all_param = 0
    for _, param in self.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_bytes = (
                param.quant_storage.itemsize if hasattr(param, "quant_storage") else 1
            )
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def print_trainable_parameters(self: nn.Module) -> None:
    """
    Prints the number of trainable parameters in the model.

    Note: print_trainable_parameters() uses get_nb_trainable_parameters() which is different from
    num_parameters(only_trainable=True) from huggingface/transformers. get_nb_trainable_parameters() returns
    (trainable parameters, all parameters) of the Peft Model which includes modified backbone transformer model.
    For techniques like LoRA, the backbone transformer model is modified in place with LoRA modules. However, for
    prompt tuning, the backbone transformer model is unmodified. num_parameters(only_trainable=True) returns number
    of trainable parameters of the backbone transformer model which can be different.
    """
    trainable_params, all_param = get_nb_trainable_parameters(self)

    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )
