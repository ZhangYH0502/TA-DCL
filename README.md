# TA-DCL：This is a multi-label classification method for medical images.

We propose Triplet Attention and Dual-pool Contrastve Learning (TA-DCL) for multi-label medical image classification. TA-DCL architecture is a triplet attention network (TAN), which combines category-attention, self-attention and cross-attention together to learn high-quality label embeddings for all disease labels by mining effective information from medical images. DCL includes dual-pool contrastive training (DCT) and dual-pool contrastive inference (DCI). DCT optimizes the clustering centers of label embeddings belonging to different disease labels to improve the discrimination of label embeddings. DCI relieves the error classification of sick cases for reducing the clinical risk and improving the ability to detect unseen diseases by contrast of differences.

## The model architecture is shown as follows:
![Image text](https://github.com/ZhangYH0502/TA-DCL/blob/master/fig8.png)

Main File Configs: <br>
* train.py: the main file to run for training the model; <br>
* test.py: test the trained model; <br>
* patchGAN_discriminator.py: discriminator model; <br>
* basic_unet.py: generator model; <br>
* loss.py: loss function; <br>
* Dataset.py: a dataloader to read data. <br>

<br>

How to run our code: <br>
* prepare your data with the paired images; <br>
* modify the data path and reading mode in Dataset.py; <br>
* run train.py to train the model; <br>
* run test.py to test the trained model.
