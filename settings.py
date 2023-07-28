"""
Global config
"""
# number of each residual block
DRK_53_RESIDUAL_BLOCK_NUMS = [1, 2, 8, 8, 4]
# output channels of each residual block
DRK_53_LAYER_OUT_CHANNELS = [64, 128, 256, 512, 1024]
# the number of anchors in each scale
ANCHORS_NUM = 3
# class of VOC dataset
VOC_CLASS_NUM = 20
# class of COCO dataset
COCO_CLASS_NUM = 80
# anchors obtained based on clustering algorithm using a distance of 1 - iou(anchors, gt_box)
ANCHORS = [[(116, 90), (156, 198), (327, 326)],
           [(30, 61), (64, 45), (59, 119)],
           [(10, 13), (16, 30), (33, 23)], ]
