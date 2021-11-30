from __future__ import print_function

FEATURE_AGNOSTIC = 1
FEATURE_DEPENDENT = 0


class obj(object):
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, (list, tuple)):
               setattr(self, key, [obj(x) if isinstance(x, dict) else x for x in value])
            else:
               setattr(self, key, obj(value) if isinstance(value, dict) else value)


config = {
    # training
    "lr": 0.0001,
    "train_warper": False,
    "simplified_training": True,
    # network
    "downsample_scale": 1,
    "mode": "training",  # for evaluation/ submission, change this to evaluation.
    "network_downsample_scale": 32,  # for resnet34 is 32
    # feature
    "small_window": True,
    "use_avg_pooling": False,
    "feature_level_start": 0,
    "feature_level_end": 2,
    "warper_type": FEATURE_DEPENDENT,
    "scorer_n_neurons": 32,
    "use_handcraft_score": True,
    "use_bilinear_scorer": False,
    # PM
    "PM_iterations": 3,
    "max_surf_normal": 50.,
    "min_surf_normal": -50.,
    "min_depth": 0.,
    "max_depth": 200.,
    "min_near_depth_pert": 1,
    "min_near_normal_pert": 5.,
    "max_near_depth_pert": 2.,
    "max_near_normal_pert": 10.,
    "max_far_depth_pert": 20.,
    "max_far_normal_pert": 50.,
}

config = obj(config)
