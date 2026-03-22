import torch
import clip

def text_prompt(data):
    # text_aug = ['{}']
    text_aug = 'This is a video about {}'
    # data.classes: [num_classes, 77]
    classes = torch.cat([clip.tokenize(text_aug.format(c)) for i, c in data.classes])
    return classes

def text_prompt_ensemble(data):
    # text_aug = ['{}']
    text_dict = {}
    text_aug = [
                f'a photo of {{}}.',
                f'a photo of a person {{}}.',
                f'a photo of a person using {{}}.',
                f'a photo of a person doing {{}}.',
                f'a photo of a person during {{}}.',
                f'a photo of a person performing {{}}.',
                f'a photo of a person practicing {{}}.',
                f'a video of {{}}.',
                f'a video of a person {{}}.',
                f'a video of a person using {{}}.',
                f'a video of a person doing {{}}.',
                f'a video of a person during {{}}.',
                f'a video of a person performing {{}}.',
                f'a video of a person practicing {{}}.',
                f'a example of {{}}.',
                f'a example of a person {{}}.',
                f'a example of a person using {{}}.',
                f'a example of a person doing {{}}.',
                f'a example of a person during {{}}.',
                f'a example of a person performing {{}}.',
                f'a example of a person practicing {{}}.',
                f'a demonstration of {{}}.',
                f'a demonstration of a person {{}}.',
                f'a demonstration of a person using {{}}.',
                f'a demonstration of a person doing {{}}.',
                f'a demonstration of a person during {{}}.',
                f'a demonstration of a person performing {{}}.',
                f'a demonstration of a person practicing {{}}.',           
            ]
            
    # text_aug = [
    #             f'A video of a person {{}}.',          
    #         ]

    # text_aug = [
    #             f'This is a video about {{}}',          
    #         ]
            
    # data.classes: [num_classes, 77]
    for idx, template in enumerate(text_aug):
        # print('11', [template.format(c) for i, c in data.classes])
        text_dict[idx] = torch.cat([clip.tokenize(template.format(c)) for i, c in data.classes])
    # classes = torch.cat([clip.tokenize(text_aug.format(c)) for i, c in data.classes])
    return text_dict


def text_prompt_ensemble_for_crime(data):
    # text_aug = ['{}']
    text_dict = {}
    text_aug = [
        'a video of {}.',
        'a video showing an event of {}.',
        'a surveillance video capturing an act of {}.',
        'a CCTV recording of {}.',
        'a scene depicting {}.',
        'a person is committing an act of {}.', # 对于异常类很有效
        'a person is involved in {}.',
        'this is a recording of an event of {}.',
        'a photo of {}.', # 加入一些图片模板可以增加多样性
        'an example of {}.'
    ]
            
    # text_aug = [
    #             f'A video of a person {{}}.',          
    #         ]

    # text_aug = [
    #             f'This is a video about {{}}',          
    #         ]
            
    # data.classes: [num_classes, 77]
    for idx, template in enumerate(text_aug):
        # print('11', [template.format(c) for i, c in data.classes])
        text_dict[idx] = torch.cat([clip.tokenize(template.format(c)) for i, c in data.classes])
    # classes = torch.cat([clip.tokenize(text_aug.format(c)) for i, c in data.classes])
    return text_dict

def text_prompt_ensemble_viclip(data):
    # text_aug = ['{}']
    text_dict = {}
    text_aug = [
                f'a photo of {{}}.',
                f'a photo of a person {{}}.',
                f'a photo of a person using {{}}.',
                f'a photo of a person doing {{}}.',
                f'a photo of a person during {{}}.',
                f'a photo of a person performing {{}}.',
                f'a photo of a person practicing {{}}.',
                f'a video of {{}}.',
                f'a video of a person {{}}.',
                f'a video of a person using {{}}.',
                f'a video of a person doing {{}}.',
                f'a video of a person during {{}}.',
                f'a video of a person performing {{}}.',
                f'a video of a person practicing {{}}.',
                f'a example of {{}}.',
                f'a example of a person {{}}.',
                f'a example of a person using {{}}.',
                f'a example of a person doing {{}}.',
                f'a example of a person during {{}}.',
                f'a example of a person performing {{}}.',
                f'a example of a person practicing {{}}.',
                f'a demonstration of {{}}.',
                f'a demonstration of a person {{}}.',
                f'a demonstration of a person using {{}}.',
                f'a demonstration of a person doing {{}}.',
                f'a demonstration of a person during {{}}.',
                f'a demonstration of a person performing {{}}.',
                f'a demonstration of a person practicing {{}}.',           
            ]
            
    # text_aug = [
    #             f'A video of a person {{}}.',          
    #         ]

    # text_aug = [
    #             f'This is a video about {{}}',          
    #         ]
            
    # data.classes: [num_classes, 77]
    for idx, template in enumerate(text_aug):
        # print('11', [template.format(c) for i, c in data.classes])
        text_dict[idx] = [template.format(c) for i, c in data.classes]
    # classes = torch.cat([clip.tokenize(text_aug.format(c)) for i, c in data.classes])
    return text_dict

def text_prompt_ensemble_for_ssv2(data):
    """
    Text prompt ensemble specifically designed for Something-Something V2 dataset.
    SSv2 focuses on temporal reasoning and object interactions with detailed action descriptions.
    """
    text_dict = {}
    text_aug = [
        'a video of {}.',
        'a person is {}.',
        'someone is {}.',
        'a demonstration of {}.',
        'an example of {}.',
        'a clip showing {}.',
        'a recording of {}.',
        'this is a video about {}.',
        'a scene of {}.',
        'footage of {}.',
        'a person performing the action of {}.',
        'a video demonstrating {}.',
        'a short clip of {}.',
        'someone performing {}.',
        'a video clip showing {}.',
        'an action of {}.',
        'a temporal sequence of {}.',
        'a person interacting with something by {}.',
        'someone manipulating objects by {}.',
        'a hand-object interaction of {}.',
    ]
            
    # data.classes: [num_classes, 77]
    for idx, template in enumerate(text_aug):
        text_dict[idx] = torch.cat([clip.tokenize(template.format(c)) for i, c in data.classes])
    
    return text_dict