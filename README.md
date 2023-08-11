# EyeLearn

The dataset and code for the paper entitled *Artifact-Tolerant Clustering-Guided Contrastive Embedding Learning for Ophthalmic Images* published in the IEEE Journal of Biomedical and Health Informatics. Here is the [link](https://ieeexplore.ieee.org/document/10159482) for the paper.

<img src="imgs/Fig1.png" width="700">

## Requirements
Python 3.8 <br/>
tensorflow 2.4.0 <br/>
opencv-python 4.5.5

## Dataset
The dataset includes 500 OCT retinal nerve fiber layer thickness (RNFLT) maps (dimension 225 x 225) from 500 unique glaucoma patients. The glaucoma label and visual field mean deviation (MD) information are also included in the data. The dataset can be accessed via this [link](https://ophai.hms.harvard.edu/datasets/harvard-glaucoma-detection-progression-1000-samples/). This dataset can only be used for non-commercial research purposes. At no time, the dataset shall be used for clinical decisions or patient care. The data use license is [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

Here are sample codes to visualize the RNFLT map:
````
from utils.map_handler import *

rnflts = np.load('dataset/rnflt_map500.npy')
plot_2dmap(rnflts[0], show_cup=True)
````
<img src="imgs/Fig2.png" width="250">

## Pretrained Model
EyeLearn_weights.72-0.0019.h5 [BaiduDisk](https://pan.baidu.com/s/1cX8t3OHLCpVb7HI0AumQqA?pwd=xqbt)


## Use the Model
````
from models import rnflt2vec

# load the pretrained model
eyelearn = rnflt2vec.construct_model_from_args(args)
model.load('EyeLearn_weights.72-0.0019.h5', train_bn=False, lr=0.00005)

# embedding learning model
encoder = eyelearn.model.get_layer('embed_model')
model_embed = Model(inputs=encoder.inputs, 
                    outputs=encoder.get_layer('encoder_output').output)
                    
# artifact correction model                   
model_correction = Model(inputs=[eyelearn.model.inputs[0], eyelearn.model.inputs[1]],
                                 outputs=eyelearn.model.output[0])
                                 
# embedding inference
embeds = model_embed.predict([masked_maps, masks]) 
# artifact correction
preds = model_inpaint.predict([masked_maps, masks]) 
````

#### Artifact correction examples: <br />
<img src="imgs/example.png" width="800">

## Citation
Shi, M., Lokhande, A., Fazli, M.S., Sharma, V., Tian, Y., Luo, Y., Pasquale, L.R., Elze, T., Boland, M.V., Zebardast, N. Friedman, D.S., and Wang M., 2022. Artifact-Tolerant Clustering-Guided Contrastive Embedding Learning for Ophthalmic Images. arXiv preprint arXiv:2209.00773.
