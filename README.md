# Real-Time Polyp Detection, Localisation and Segmentation in Colonoscopy Using Deep Learning
# ColonSegNet


ColonSegNet is an encoder-decoder that uses residual block with squeeze and excitation network as the main component. The network is designed to have very few trainable parameters as compared to other networks baseline networks such as U-Net, PSPNet, DeepLabV3+, and others.  The use of fewer trainable parameters makes the proposed architecture a very light-weight network that leads to real-time performance.

[Real-Time Polyp Detection, Localisation and Segmentation in Colonoscopy Using Deep Learning](access.pdf).

## Architecture
<img src="ColonSegNet.png" align="center">

## Requirements:
	os
	numpy
	cv2
	tensorflow
	glob
	tqdm

## Folders:
	data: Contains the set of three dataset as mentioned.
	files: Contains the csv file and weight file generated during training.
	new_data: Contains two subfolder `images` and `masks`, they contains the augmented images and masks.

## Files:
	1. process_image.py: Augment the images and mask for the training dataset.
	2. data_generator.py: Dataset generator for the keras.
	3. infer.py: Run your model on test dataset and all the result are saved in the result` folder. The images are in the sequence: Image,Ground Truth Mask, Predicted Mask.
	4. run.py: Train the unet.
	5. unet.py: Contains the code for building the UNet architecture.
	6. resunet.py: Contains the code for building the ResUNet architecture.
	7. m_resunet.py: Contains the code for building the ResUNet++ architecture.
	8. mertrics.py: Contains the code for dice coefficient metric and dice coefficient loss. 


## First check for the correct path and the patameters.
1.	python3 process_image.py - to augment training dataset.
2.	python3 run.py - to train the model.
3.	python3 infer.py - to test and generate the mask.



## Results

<img src="qualitative_results.png">
<img src="quantitative.png">

## Citation
Please cite our paper if you find the work useful: 
<pre>
@article{jha2020real,
  title={Real-Time Polyp Detection, Localisation and Segmentation in Colonoscopy Using Deep Learning},
  author={Jha, Debesh and Ali, Sharib and Johansen, H{\aa}vard D and Johansen, Dag D and Rittscher, Jens and Riegler, Michael A and Halvorsen, P{\aa}l},
  journal={IEEE Access},
  year={2021}
}
</pre>

## Contact
Please contact debesh@simula.no for any further questions.
