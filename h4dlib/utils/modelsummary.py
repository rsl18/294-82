"""
      code by Tae Hwan Jung(Jeff Jung) @graykode
      code reference : https://github.com/pytorch/pytorch/issues/2001,
                        https://gist.github.com/HTLife/b6640af9d6e7d765411f8aa9aa94b837,
                        https://github.com/sksq96/pytorch-summary
      Inspired by https://github.com/sksq96/pytorch-summary
      But 'torchsummary' module only works in Vision Network model So I fixed it!
"""

import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from functools import reduce

from torch.nn.modules.module import _addindent


def hierarchicalsummary(model):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if p is not None:
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        main_str += ", {:,} params".format(total_params)
        return main_str, total_params

    string, count = repr(model)
    print(string)
    return count


def summary(model, *inputs, batch_size=-1, show_input=True, show_hierarchical=True):
    if show_hierarchical is True:
        hierarchicalsummary(model)

    def register_hook(module):
        def hook(module, input, output=None):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()

            if len(input) != 0:
                if type(input[0])==tuple:
                    summary[m_key]["multi_out"] = True
                    summary[m_key]["sub_key"] = "a"
                    # for i, e in enumerate(input[0]):
                    #     sub_key = f"{class_name}-{module_idx+1}-{i}"
                    #     summary[sub_key] = OrderedDict()
                    #     summary[sub_key]["input_shape"] = list(e.size())
                    #     summary[sub_key]["input_shape"][0] = batch_size
                else:
                    summary[m_key]["input_shape"] = list(input[0].size())
                    summary[m_key]["input_shape"][0] = batch_size
            else:
                summary[m_key]["input_shape"] = input

            if show_input is False and output is not None:
                # print("key: " , m_key)
                if isinstance(output, (list, tuple)):
                    for out in output:
                        if isinstance(out, torch.Tensor):
                            summary[m_key]["output_shape"] = [
                                [-1] + list(out.size())[1:]
                            ][0]
                        elif isinstance(out, dict):
                            print(out.keys())
                        else:
                            summary[m_key]["output_shape"] = [
                                [-1] + list(out[0].size())[1:]
                            ][0]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            if show_input is True:
                hooks.append(module.register_forward_pre_hook(hook))
            else:
                hooks.append(module.register_forward_hook(hook))
        # else:
        #     print("no hoook", module, type(module))
    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)
    model(*inputs)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("-----------------------------------------------------------------------")
    if show_input is True:
        line_new = "{:>25}  {:>25} {:>15}".format(
            "Layer (type)", "Input Shape", "Param #"
        )
    else:
        line_new = "{:>25}  {:>25} {:>15}".format(
            "Layer (type)", "Output Shape", "Param #"
        )
    print(line_new)
    print("=======================================================================")

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        nb_params = summary[layer]["nb_params"] if "nb_params" in summary[layer] else 0
        # print("Layer: ", layer)
        if "multi_out" not in summary[layer]:
            # input_shape, output_shape, trainable, nb_params
            if show_input is True:
                line_new = "{:>25}  {:>25} {:>15}".format(
                    layer,
                    str(summary[layer]["input_shape"]),
                    "{0:,}".format(nb_params),
                )
            else:
                line_new = "{:>25}  {:>25} {:>15}".format(
                    layer,
                    str(summary[layer]["output_shape"]),
                    "{0:,}".format(nb_params),
                )

            total_params += nb_params
            if show_input is True:
                total_output += np.prod(summary[layer]["input_shape"])
            else:
                total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += nb_params

            print(line_new)

    print("=======================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("-----------------------------------------------------------------------")
