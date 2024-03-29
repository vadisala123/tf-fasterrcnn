# Tensorflow Faster RCNN
This repo was cloned from https://github.com/kbardool/keras-frcnn and modified to run with
Tensorflow 2.0+ & Python 3. The usage of Keras was changed to Tensorflow. The changes I have done
to original repo is for educational purpose only.  

USAGE:
- Both theano and tensorflow backends are supported. However compile times are very high in
 theano, and tensorflow is highly recommended.
 
- the Pascal VOC data set (images and annotations for bounding boxes around the classified
 objects) can be obtained from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May
 -2012.tar
- simple_parser.py provides an alternative way to input data, using a text file. Simply provide a
 text file, with each
line containing:

    `filepath,x1,y1,x2,y2,class_name`

    For example:

    `/data/imgs/img_001.jpg,837,346,981,456,cow
    /data/imgs/img_002.jpg,215,312,279,391,cat
    `
    
    The classes will be inferred from the file. To use the simple parser instead of the default
     pascal voc style parser,
    use the command line option `-o simple`. For example `python train_frcnn_v2.py -o simple -p
     my_data.txt`.
     
- Please use `train_frcnn_v2.py` to train the model as it is efficient and runs faster than
 `train_frcnn.py`. To train on Pascal VOC data, simply do `python train_frcnn_v2.py -o simple -p
  train_annotate.txt -v val_annotate.txt`. `train_annotate.txt` and `val_annotate.txt` are
   training and validation files created using the below format.

- Running `train_frcnn_v2.py` or `train_frcnn.py` will write weights to disk to an hdf5 file, as
 well as all the setting of the training run to a `pickle` file. These settings can then be
  loaded by `test_frcnn.py` for any testing.

- `test_frcnn.py` can be used to perform inference, given pre-trained weights and a config file
. Specify a path to the folder containing images and model weights: `python test_frcnn.py -p
 /path/to/test_data --model_weights_path model_frcnn_0009.hdf5`
- Data augmentation can be applied by specifying `--hf` for horizontal flips, `--vf` for vertical
 flips and `--rot` for 90 degree rotations


NOTES:
- config.py contains all settings for the train or test run. The default settings match those in
 the original Faster-RCNN paper. The anchor box sizes are [128, 256, 512] and the ratios are [1:1
 , 1:2, 2:1].
- The theano backend by default uses a 7x7 pooling region, instead of 14x14 as in the frcnn paper
. This cuts down compiling time slightly.
- The tensorflow backend performs a resize on the pooling region, instead of max pooling. This is
 much more efficient and has little impact on results.
- Batch size is limited to 1.


Example output:

![ex1](http://i.imgur.com/7Lmb2RC.png)
![ex2](http://i.imgur.com/h58kCIV.png)
![ex3](http://i.imgur.com/EbvGBaG.png)
![ex4](http://i.imgur.com/i5UAgLb.png)

ISSUES:

- If you run out of memory, try reducing the number of ROIs that are processed simultaneously
. Try passing a lower `-n` to `train_frcnn.py`. Alternatively, try reducing the image size from
 the default value of 600 (this setting is found in `config.py`.
