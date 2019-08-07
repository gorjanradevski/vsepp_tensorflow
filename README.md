# Improving Visual-Semantic Embeddings with Hard Negatives

Tensorflow implementation of [Improving Visual-Semantic Embeddings with Hard Negatives](https://arxiv.org/abs/1707.05612) based on the [original implementation](https://github.com/fartashf/vsepp)
in PyTorch.

### Download the data 

The data should be downloaded in the ```/data``` directory. To download the ```Flickr8k``` dataset, run:

```
!wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
!wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
!unzip -qq Flickr8k_Dataset.zip
!unzip -qq Flickr8k_text.zip
!rm Flickr8k_Dataset.zip
!rm Flickr8k_text.zip
```

or to download the ```Flickr30k``` dataset run:

```
!wget https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/train.txt
!wget https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/val.txt
!wget https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/test.txt
!wget http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k-images.tar
!wget http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.tar.gz
!tar -xf flickr30k-images.tar
!tar -xzf flickr30k.tar.gz
```

### Training a model (Flickr8k example)

To train a model, the ```train.py``` script needs to be run in the following way:

```
python src/train.py --batch_size 128\
                    --prefetch_size 5\
                    --images_path "data/Flicker8k_Dataset/"\
                    --texts_path "data/Flickr8k.token.txt"\
                    --train_imgs_file_path "data/Flickr_8k.trainImages.txt"\
                    --val_imgs_file_path "data/Flickr_8k.devImages.txt"\
                    --save_model_path "models/name_of_the_model"\
```

Where the default values are taken for the other command line arguments.

### Performing inference (Flickr8k example)

If there is a trained model instance, inference on the test set can be run in the following way:

```
python src/inference.py --batch_size 128\
                        --prefetch_size 5\
                        --images_path "data/Flicker8k_Dataset/"\
                        --texts_path "data/Flickr8k.token.txt"\
                        --test_imgs_file_path "data/Flickr_8k.trainImages.txt"\
                        --checkpointh_path "models/name_of_the_model"\
                        --joint_space 1024 # Assuming that the joint space of the trained instance is 1024\
                        --num_layers 1 # Assuming that the number of layers of the trained instance is 1
```