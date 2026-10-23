import json
from mmengine.dist import get_dist_info
from mmengine.optim import DefaultOptimWrapperConstructor
from mmdet.registry import OPTIM_WRAPPER_CONSTRUCTORS


def get_num_layer_for_vit(var_name, num_max_layer):
    # tokens / pos_embed / patch_embed
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed"):
        return 0
    if var_name.startswith("backbone.patch_embed"):
        return 0

    if var_name.startswith("backbone.blocks."):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1

    return num_max_layer - 1


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LayerDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    def add_params(self, params, module, prefix='', **kwargs):
        parameter_groups = {}

        num_blocks = self.paramwise_cfg.get('num_blocks') + 2
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        weight_decay = self.base_wd

        print(self.paramwise_cfg)
        print(f"Build LayerDecayOptimizerConstructor {layer_decay_rate:.6f} - {num_blocks}")

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            
            short_name = name.split('.')[-1]

            if len(param.shape) == 1 or name.endswith(".bias") or short_name in ('pos_embed', 'cls_token'):
                group_decay = 0.0
                group_name = "no_decay"
            else:
                group_decay = weight_decay
                group_name = "decay"

            layer_id = get_num_layer_for_vit(name, num_blocks)
            full_group_name = f"layer_{layer_id}_{group_name}"

            if full_group_name not in parameter_groups:
                scale = layer_decay_rate ** (num_blocks - layer_id - 1)
                parameter_groups[full_group_name] = dict(
                    weight_decay=group_decay,
                    params=[],
                    param_names=[],
                    lr_scale=scale,
                    group_name=full_group_name,
                    lr=scale * self.base_lr,
                )

            parameter_groups[full_group_name]["params"].append(param)
            parameter_groups[full_group_name]["param_names"].append(name)

        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {
                k: dict(
                    lr_scale=v["lr_scale"],
                    lr=v["lr"],
                    weight_decay=v["weight_decay"],
                    param_names=v["param_names"],
                )
                for k, v in parameter_groups.items()
            }
            print("Param groups =", json.dumps(to_display, indent=2))

        params.extend(parameter_groups.values())