# PyTorch Implementation of Neural Style Transfer

Implementation of the [neural style transfer algorithm](https://arxiv.org/abs/1508.06576) (Gatys, Ecker, Bethge) using [PyTorch](https://pytorch.org/). 

Generally follows the PyTorch [tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html), with some modifications.

Style transfer can run either on a single content image, or a directory of content images.

## Example
Style image + Content image -> Result

<img src='https://i.imgur.com/a6ymELU.jpg' width='250' height='250'><img src='http://i.imgur.com/5qILgh1.jpg' width='250' height='250'><img src='https://i.imgur.com/Kr7oeVV.jpg' width='250' height='250'>

## Setup
Create and activate a new virtual environment:
```
$ conda create -n style_transfer python=3
$ conda activate style_transfer
```

Install dependencies:
```
$ pip install -r requirements.txt
```

## File Overview 
- `single_neural_style_transfer.py`: Run neural style transfer on a single content image.
- `batch_neural_style_transfer.py`: Run neural style transfer on a directory of content images.
- `requirements.txt`: Contains dependencies for project to be installed.

Optional:
- `style_imgs/`: Directory containing images I thought would be nice to use as style input.
- `content_images/`: Directory containing images I thought would be nice to use as content input.
- `results/`: Directory containing results from previous runs.

## Usage 

### Neural style transfer on a single image
In `single_neural_style_transfer.py`: 
- Setting the style image
  - On line 258: `style_path = 'style_imgs/marbled_paint_1.jpeg'`
  - Replace this path with the path to the image you want to use as style input.

- Setting the content image
  - On line 259: `content_path = 'content_imgs/clouds/clouds12.jpg'`
  - Replace this path with the path to the image you want to use as content input.
  
- Setting the output path
  - On line 275: `torchvision.utils.save_image(output, 'output.png')`
  - By default, the result will save to a file named `output.png` in the same directory. 
  - If you want, replace this with the filepath where you would like the output image to go.

### Neural style transfer on a directory of images
In `batch_neural_style_transfer.py`: 
- Setting the style image 
  - On line 249: `style_path = 'style_imgs/turner_sunset.jpg'`
  - Replace this path with the path to the image you want to use as style input.
- Setting the content image directory 
  - On line 250: `content_paths = 'content_imgs/*/*'`
  - Replace this path with the directory of images you'd like to use as content input.
  - `content_images/*/*` will grab nested directories in the `content_images` directory. `content_images/*` will grab all images inside the `content_images` directory (unnested).
  
- Setting the output path
  - On line 251: `output_folder = 'results/turner_sunset/'`
  - Replace this with the directory you would like the output image results to go.


### Other options
- Image size: `imsize = 256`
  - By default, the resulting images will be 256 x 256 pixels.  
- Style and content weights, number of iterations
```
def run_style_transfer(cnn, norm_mean, norm_std,
        content_img, style_img, input_img,
        num_steps=300, style_weight=1000000, content_weight=1):
```
  - Can tweak style_weight, content_weight, and num_steps.
  - More steps -> more stylized. 
