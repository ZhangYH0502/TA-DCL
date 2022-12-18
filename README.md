# TA-DCL：This is a multi-label classification method for medical images.

We propose Triplet Attention and Dual-pool Contrastve Learning (TA-DCL) for multi-label medical image classification. TA-DCL architecture is a triplet attention network (TAN), which combines category-attention, self-attention and cross-attention together to learn high-quality label embeddings for all disease labels by mining effective information from medical images. DCL includes dual-pool contrastive training (DCT) and dual-pool contrastive inference (DCI). DCT optimizes the clustering centers of label embeddings belonging to different disease labels to improve the discrimination of label embeddings. DCI relieves the error classification of sick cases for reducing the clinical risk and improving the ability to detect unseen diseases by contrast of differences.

## The model architecture is shown as follows:
![Image text](https://github.com/ZhangYH0502/TA-DCL/blob/master/fig8.png)

<br>

How to run our code: <br>
* prepare your data and divide them into sick pool and divide pool; <br>
* modify the data path and reading mode in dataloaders; <br>
* modify the model attributes in config_args.py; <br>
* run main.py to train and test the model.
