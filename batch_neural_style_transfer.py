from __future__ import print_function

import copy
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

import torchvision

import torchvision.transforms as transforms
import torchvision.models as models

### Setup
# Set GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set output image size
imsize = 512

loader = transforms.Compose([
    transforms.Resize(imsize), # Resize images
    transforms.ToTensor()]) # Convert to torch tensor

def image_loader(image_name):
    image = Image.open(image_name)
    image = image.resize((imsize,imsize)) # Resize
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


### Loss Functions

# Content loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# Style loss
def gram_matrix(input):
    a, b, c, d = input.size()
    # a: batch size(=1)
    # b: number of feature maps
    # (c,d): dimensions of a f. map (N=c*d)

    features = input.view(a*b, c*d) # Reshape to transpose F_xl

    # Matrix multiplication to compute the gram product
    G = torch.mm(features, features.t())

    # Normalize gram matrix values by dividing by # elements in each feature map
    # Important bc style features tend to be in deeper network layers
    G_norm = G.div(a*b*c*d)

    return G_norm

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


## Normalize input image w/ module so we can put it in a nn.Sequential layer
class Normalization(nn.Module):
    def __init__(self, mean, std):
        """
        Reshape mean and std into [C x 1 x 1] to work w/ image Tensor of shape
        [B x C x H x W]
        B: batch size
        C: # of channels
        H: height
        W: width
        """
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # Normalize image
        img_norm = (img - self.mean) / self.std
        return img_norm


### Network Architecture
# Set which layers we want to compute style and content losses at
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def create_network_with_losses(cnn, norm_mean, norm_std,
        style_img, content_img,
        content_layers=content_layers_default,
        style_layers=style_layers_default):
    """
    Sequential module contains ordered list of child modules, in order of depth.
    e.g. vgg19.features contains (Conv2d, ReLU, MaxPool2d, Conv2d, ReLU...)

    Want to add content and style loss layers after the convolution layer they
    are detecting by creating new Sequential module w/ content loss and style loss
    modules correctly inserted.
    """
    print('Creating network...')

    cnn = copy.deepcopy(cnn)

    # Normalize
    normalization = Normalization(norm_mean, norm_std).to(device)

    # Keep track of losses
    content_losses = []
    style_losses = []

    # Assuming cnn is nn.Sequential, make new Sequential layer to add
    model = nn.Sequential(normalization)

    i = 0 # Increment for each convolutional layer
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            # 2D convolutional layer
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            # rectified linear unit layer
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False) # Replace w/ out-of-place ReLU
        elif isinstance(layer, nn.MaxPool2d):
            # 2d max pooling layer
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            # batch normalization layer
            name = 'bnorm_{}'.format(i)
        else:
            layer_name = layer.__class__.__name__
            raise RuntimeError('Unrecognized layer: {}'.format(layer_name))

        model.add_module(name, layer)

        # Add content loss layer to network if current layer is a content layer
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module('content_loss_{}'.format(i), content_loss)
            content_losses.append(content_loss)

        # Add style loss layer if current layer is a style layer
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module('style_loss_{}'.format(i), style_loss)
            style_losses.append(style_loss)


    # Trim off layers after the last content and style losses
    for i in range(len(model)-1, -1, -1):
        current_layer = model[i]
        if isinstance(current_layer, ContentLoss) or isinstance(current_layer, StyleLoss):
            break

    model = model[:(i+1)]

    return model, style_losses, content_losses


### Gradient Descent
def get_input_optimizer(input_image):
    """
    Set optimizer to use the Limited-memory BFGS optimization algorithm
    """
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


### Neural Style Transfer
def run_style_transfer(cnn, norm_mean, norm_std,
        content_img, style_img, input_img,
        num_steps=300, style_weight=1000000, content_weight=1):
    """ Run the style transfer. """
    print('Building style transfer model...')
    model, style_losses, content_losses = create_network_with_losses(cnn, norm_mean, norm_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    run = [0]

    while run[0] <= num_steps:
        def closure():
            # Correct values of updated input image by clamping
            input_img.data.clamp_(0,1)

            optimizer.zero_grad() # Reset gradients to zero for backward pass
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            # Weight style and content scores
            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print('Run {}:'.format(run))
                print('Style loss: {:4f}'.format(style_score.item()))
                print('Content loss: {:4f}'.format(content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

        # Clamp one last time
        input_img.data.clamp_(0,1)

        return input_img


### Run algorithm
# Import model
"""
Use pretrained, 19-layer VGG network.
Use features module and set to evaluation mode.
"""
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# Normalize images w/ mean and std
cnn_norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Set style image, content folder, and output folder paths
style_path = 'style_imgs/whistler_nocturne_dark.jpg'
content_paths = 'content_imgs/clouds/*'
output_folder = 'results/whistler_clouds/'

style_img = image_loader(style_path)

count = 0
for content_path in glob.glob(content_paths):

    # Set content and result paths
    output_path = output_folder + 'result_{}.jpg'.format(count)
    content_img = image_loader(content_path)

    # Set input image: can use white noise, or a copy of the input image
    input_img = content_img.clone() # Input image copy

     # Check style and content images are same size
    assert style_img.size() == content_img.size(), 'Style and content images need to be the same size.'

    ### Run style transfer
    output = run_style_transfer(cnn, cnn_norm_mean, cnn_norm_std, content_img, style_img, input_img)

    # Write output image to file
    print('Writing output to file: ', output_path)
    torchvision.utils.save_image(output, output_path)

    count += 1

