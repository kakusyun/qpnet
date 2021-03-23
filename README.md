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
