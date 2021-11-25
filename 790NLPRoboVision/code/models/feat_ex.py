import torch
import torch.nn.functional as F
from torchvision.models import resnet50
# from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
# from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


class FeatEx(torch.nn.Module):
    def __init__(self):
        super(FeatEx, self).__init__()
        # Get a resnet50 backbone
        m = resnet50()
        # Extract 4 main layers (note: MaskRCNN needs this particular name
        # mapping for return nodes)
        self.body = create_feature_extractor(
            m, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2])})
        # Dry run to get number of channels for FPN
        inp = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = self.body(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        # Build FPN
        self.out_channels = 64
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool())

    def forward(self, x):
        bs, c, h, w = x.shape
        x = self.body(x)
        x = self.fpn(x)

        feat_1 = F.interpolate(x['0'], size=(h, w), mode='bilinear', align_corners=False)
        feat_2 = F.interpolate(x['1'], size=(h, w), mode='bilinear', align_corners=False)
        # feat_3 = F.interpolate(x['2'], size=(h, w), mode='bilinear', align_corners=False)
        # feat_4 = F.interpolate(x['3'], size=(h, w), mode='bilinear', align_corners=False)
        output = torch.cat([feat_1, feat_2], dim=1)  # 1/16

        return output
