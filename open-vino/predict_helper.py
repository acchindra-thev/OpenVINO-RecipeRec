import torch
from PIL import Image
import numpy as np
from torchvision import models
import pandas as pd

def load_checkpoint(path):
    checkpoint = torch.load(path)

    # load the model
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    
    model.load_state_dict(checkpoint['state_dict'])
    
    model.idx_to_class = checkpoint['idx_to_class']

    return model

def process_image(image):
    image = Image.open(image)
    
    # Get original dimensions
    w, h = image.size

    # Find longer side to crop 
    if w < h:
        resize_size=(256,256**2)
    else: 
        resize_size=(256**2, 256)
 
    image.thumbnail(resize_size)
    
    # crop
    crop_size = 224
    width, height = image.size   # Get dimensions
    
    left = (width - crop_size)/2
    top = (height - crop_size)/2
    right = (width + crop_size)/2
    bottom = (height + crop_size)/2
    
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image)
    
    # normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = np.array(image) / 255
    
    image = (image_array - mean) / std
    
    # reorder dimensions for color channel
    image = image.transpose((2,0,1))
    
    return torch.from_numpy(image)

def predict(image_path, model, topk, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model. '''
    if gpu:
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    model.to(device)
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    image = image.to(device).float()
    
    model.eval()
    with torch.no_grad():
        
        logps = model.forward(image)
        
        ps = torch.exp(logps)
        
        top_p, top_class = ps.topk(topk)

    model.train()
    
    class_names = []
    for indexes in top_class.cpu().numpy()[0]:
        class_names.append(model.idx_to_class[indexes])


    probs = top_p.tolist()[0]
    
    return probs,class_names

def image_predict(image_path, model, topk=2, gpu=False):
    #Get the predictions
    probs, class_names = predict(image_path, model, topk, gpu=gpu)

    #Create DF of prediction
    top_preds = pd.DataFrame({'class': class_names, 'probability': probs})

    # Print the top predictions
    print(f"\nImage Path:\n\t{image_path}")
    print(top_preds.to_string(index=False))
    print('')