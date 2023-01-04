import torch
from collections import OrderedDict

model = torch.load('/vol/vssp/facer2vm/people/tianyang/Codes/DFAT/experiments/siammask_r50_l3_q/checkpoint_e0.pth', map_location=lambda storage, loc: storage)

new_model = OrderedDict()

for k, v in model['state_dict'].items():
    if k.startswith('features.features'):
        k = k.replace('features.features', 'backbone')
    elif k.startswith('features'):
        k = k.replace('features', 'neck')
    elif k.startswith('rpn_model'):
        k = k.replace('rpn_model', 'rpn_head')
    elif k.startswith('mask_model'):
        k = k.replace('mask_model.mask', 'mask_head')
    elif k.startswith('refine_model'):
        k = k.replace('refine_model', 'refine_head')
    new_model[k] = v

torch.save(new_model, '/vol/vssp/facer2vm/people/tianyang/Codes/DFAT/experiments/siammask_r50_l3_q/model.pth')
