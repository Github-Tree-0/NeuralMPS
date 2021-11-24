# NeuralMPS

## Download test data
Please download test data via: https://drive.google.com/drive/folders/1fm0IUYvPOe1GzjG_bxVq-qGSuQB9d_bq?usp=sharing first.
Download the folder "test_data" to the main folder "NeuralMPS", unzip it and use the name "test_data".

## Dependencies
```shell
$ conda install pytorch torchvision cudatoolkit=11.3 -c pytorch -c conda-forge
$ pip install scipy matplotlib imageio scikit-image
```

## Run the test code

Simply run

```shell
$ python test.py
```

This file will use our pretrained model to test the Sphere dataset. Results will be stored in "test_data/results", including the predicted normal map, predicted equivalent light intensity and ground truth equivalent light intensity.

## Test your own data

Please follow the data format of "test_data". Or you can refer to "datasets/my_npy_dataloader.py" to write your own dataloader.

## Declaration

Code and data of CVPR submission 3927, used for CVPR 2022 review only.
All rights reserved by the authors of the CVPR'22 submission 3927.