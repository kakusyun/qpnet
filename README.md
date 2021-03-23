Quadrant Perception Network (QPNet)
===
The implementation of the paper "A Quadrant Perception Network for Text Detection"

![image](https://github.com/kakusyun/qpnet/blob/master/images/encoding.png)


Install
===
1. Clone the project

    ```Shell
    git clone https://github.com/kakusyun/qpnet
    cd qpnet
    ```

2. Create a conda virtual environment and activate it

    ```Shell
    conda create -n qpnet python=3.7 -y
    conda activate qpnet
    ```

3. Install dependencies

    ```Shell
    # If you dont have pytorch
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch 

    pip install -r requirements.txt
    ```

4. Data preparation
   
   Please use labelme to label your samples and get the dataset like:   
   ```Shell
   # datasets/dataset_name
   |--images
   |--jsons
   ```
   If you don't use labelme, just let your data like above.
   
   ```Shell
   cd datasets/preprocessing
   python one_step_preprocessing.py
   ```
 
 5. Build
    ```Shell
    cd qpnet
    python setup.py develop
    ```

Run
===
1. Training:
    
    ```Shell
    python train.py
    ```
    
To switch single GPU training or multiply GPUs training, please change tools/train_qp.py.

2. Test:

    ```Shell
    python test.py
    ```
    
To switch single GPU training or multiply GPUs training, please change tools/test_qp.py.

3. Infer:

    ```Shell
    python infer.py
    ```
    
To switch single GPU training or multiply GPUs training, please change tools/infer_qp.py.
