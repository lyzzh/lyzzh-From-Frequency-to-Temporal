from torchsummary import summary
import torch
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.constant(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


class MaxNormDefaultConstraint(object):
    """
    Applies max L2 norm 2 to the weights until the final layer and L2 norm 0.5
    to the weights of the final layer as done in [1]_.

    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730

    """

    def apply(self, model):
        last_weight = None
        for name, module in list(model.named_children()):
            if hasattr(module, "weight") and (
                    not module.__class__.__name__.startswith("BatchNorm")
            ):
                module.weight.data = torch.renorm(
                    module.weight.data, 2, 0, maxnorm=2
                )
                last_weight = module.weight
        if last_weight is not None:
            last_weight.data = torch.renorm(last_weight.data, 2, 0, maxnorm=0.5)


class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel1_size,  **kw):
        super(SeparableConv2D, self).__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel1_size, **kw),
            # nn.BatchNorm2d(in_channels),
            nn.ELU(inplace=True),
            # pw
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), **kw),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.depth_conv(x)


def square_activation(x):
    return torch.square(x)


def safe_log(x):
    return torch.clip(torch.log(x), min=1e-7, max=1e7)




