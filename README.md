# DeepFake 101: Face Swap

This is a code repository for GradientCrescent's tutorial on the generation of a simple DeepFake, known as a Face-Swap. Based on Ovalery16's implementation of Shaoanlu's original DeepFake FaceSwap-GAN project, we've optimized the code into stand-alone modules for Google Colaboratory, aiming for maximum intutiveness, user-friendliness, and overall computational speed.

Detailed tutorial, "Into the Cageverse — Deepfaking with Autoencoders: An Implementation in Keras and Tensorflow", available on [GradientCrescent](https://medium.com/gradientcrescent/deepfaking-nicolas-cage-into-the-mcu-using-autoencoders-an-implementation-in-keras-and-tensorflow-ab47792a042f).


## Structure
The are three Ipython Notebooks in this repository, that should be run in the following order: 

	1. Data_Preprocessing.ipynb
	2. Training.ipynb
	3. Generation.ipynb

## Dataset

500 web-scraped images of Nicolas Cage, Tom Holland, and Chris Hemsworth from Google Image Search, using the "Large" image filter setting. Generally, performance improves with larger datasets, as a wider range of expressions can be covered. The use of a batch downloading addon is recommended to facilitate ease of data gathering.

## How are Deepfakes made?

Given a reference and a target set of facial images, we can train a pair of autoencoders — one for target-target reconstruction and one for reference-reference reconstruction. The autoencoder consists of an encoder and decoder component, where the autoencoder’s role is to learn to encode an input image into a lower-dimensional representation, while the decoder’s role is to reconstruct this representation back into the original image. This encoded representation contains pattern details related to the facial characteristics of the original image, such as expression.

To generate a deepfake, the encoded representation of the target is fed into the decoder of the reference, resulting in the generation of an output featuring the reference’s likeness but with the characteristics of the target.

## Results (Reference: Nicolas Cage)

</p>
<p align="center">
  <img src="https://github.com/EXJUSTICE/DeepFake_FaceSwap/blob/master/spider2.png">
</p>
                                                    
                                                    
<p align="center">
  <b>Generated outputs of Spiderman together with original input (left), after 30, 60, and 90 epochs.</b><br>
 
</p>


</p>
<p align="center">
  <img src="https://github.com/EXJUSTICE/DeepFake_FaceSwap/blob/master/thor.png">
</p>
                                                    
                                                    
<p align="center">
  <b>Generated outputs of Thor together with original input (left), after 40, 70, and 90 epochs.</b><br>
 
</p>






