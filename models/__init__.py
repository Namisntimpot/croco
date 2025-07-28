from .croco import CroCoNet, MMAE_CroCoNet
from .croco_matching import CroCoNet as CroCoNet_Matching, MMAE_CroCoNet as MMAE_CroCoNet_Matching
from .croco_downstream import CroCoDownstreamBinocular

tasks = {
    'pretrain': {
        'croco': CroCoNet,
        'mmae': MMAE_CroCoNet
    },
    'matching': {
        'croco': CroCoNet_Matching,
        'mmae': MMAE_CroCoNet_Matching,
    },
    'stereoflow': {
        'croco': CroCoDownstreamBinocular,
    }
}

def make_model(task, pretrain_type, **kwargs):
    assert task in tasks
    d = tasks[task]
    assert pretrain_type in d
    model_class = d[pretrain_type]
    return model_class(**kwargs)