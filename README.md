Quadrant Perception Network (QPNet)
===

Description: A simple and elegant model based on instance segmentation is designed for text detection in a document and natural image. Consisting of convolutional and bidirectional long short term memory (BiLSTM) networks, it focuses on segmenting the close text instances and detecting the long text to improve the practicability in real applications. The input images are encoded by their grid locations related to the four quadrants of an object and the background. BiLSTMs with transposing operations are used to combine the left-right and up-down contexts. Without bounding box regression, only one output classification branch is designed to predict the accurate location of each pixel, namely quadrant perception. Therefore, it is easy to train. Finally, simple post-processing is employed to find text locations naturally.

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
1. Train:
    
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
