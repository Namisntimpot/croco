from .croco import CroCoNet, MMAE_CroCoNet
from .croco_matching import CroCoNet as CroCoNet_Matching, MMAE_CroCoNet as MMAE_CroCoNet_Matching
from .croco_downstream import CroCoDownstreamBinocular
from .mmae import MMAEViT, CroCoViT

tasks = {
    'pretrain': {
        'croco': CroCoNet,
        'mmae': MMAE_CroCoNet,
        'croco_new_arch': CroCoViT,
        'mmae_new_arch': MMAEViT,
    },
    'matching': {
        'croco': CroCoNet_Matching,
        'mmae': MMAE_CroCoNet_Matching,
        'croco_new_arch': CroCoViT,
        'mmae_new_arch': MMAEViT,
    },
    'stereoflow': {
        'croco': CroCoDownstreamBinocular,
    }
}

def make_model(task, pretrain_type, image_size, **kwargs):
    assert task in tasks
    d = tasks[task]
    assert pretrain_type in d
    model_class = d[pretrain_type]
    return model_class(img_size=image_size, **kwargs)
    # return model_class(**kwargs)