## Introduction
This project is an implentation of the paper [<<Combining Sketch and Tone for Pencil Drawing Production>>](http://www.cse.cuhk.edu.hk/leojia/projects/pencilsketch/pencil_drawing.htm) using python. 

Now it could only produce grayscale result, the color result will be done in future.

The code reference to the an matlab implentation on github, see (https://github.com/candycat1992/PencilDrawing)   


## Dependcy
* numpy
* scipy
* opencv


## Usage
`python pencil_draw.py [image] [pencil texture img]`

eg: `python pencil_draw.py lena.jpg pencil0.jpg`

the result image will be showed

## Reference
[1]Lu C, Xu L, Jia J. Combining sketch and tone for pencil drawing production[C]//Proceedings of the Symposium on Non-Photorealistic Animation and Rendering. Eurographics Association, 2012: 65-73.
