# xpath, beautiful soup 也是指定模块的方法
# 正则表达式也是
# 树状结构，有深浅
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def is_contains(
    module_simple_name: str, module_full_name: str, modified_modules: list[str]
) -> bool:
    for requirement in modified_modules:
        if requirement in module_full_name:
            return True
    return False


def is_match(
    module_simple_name: str, module_full_name: str, modified_modules: list[str]
) -> bool:
    for requirement in modified_modules:
        if module_simple_name == requirement or (
            module_full_name.endswith(requirement) and requirement != ""
        ):
            return True
    return False


# 特殊情况，
# [""]空字符串表示根模型，也就是model本身。
# [] 空列表表示不想要任何module，返回False
def find_modules_all(model: nn.Module, modified_modules: list[str]):
    for name, module in model.named_modules():  # 先序遍历
        # print(f"name={name}, cls_name={module.__class__.__name__}")
        if is_match(module.__class__.__name__, name, modified_modules):
            yield (name, module)  # 为了让用户知道自己在干什么，筛选出来的模型对不对。
            # print("It matches!")
        # print()


def find_modules_parent_only(
    model: nn.Module, modified_modules: list[str], parent_name=""
):
    if is_match(model.__class__.__name__, parent_name, modified_modules):
        yield parent_name, model  # 一般都不是这个情况
        return  # 父模块优先，已经找到就不深究了。
    for name, children in model.named_children():  # 先序遍历
        # print(f"name={name}, cls_name={module.__class__.__name__}")
        full_name = f"{parent_name}.{name}" if parent_name != "" else name
        yield from find_modules_parent_only(children, modified_modules, full_name)
        # print("It matches!")
        # print()


find_modules = find_modules_parent_only


def find_modules_dict(model: nn.Module, modified_modules: list[str]) -> dict:
    return dict(find_modules(model, modified_modules))
