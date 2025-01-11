# CNNs_calibration
# Calibration in CNNs for Bid-Cat classification

This project explores the calibration of Convolutional Neural Networks (CNNs) for binary classification, specifically focusing on improving model reliability through Temperature Scaling. The project uses the CIFAR-10 dataset, initially training a **LeNet-5 CNN** to distinguish between birds and cats. Additionally, a **DenseNet-121** model is fine-tuned to further evaluate calibration metrics such as Expected Calibration Error (ECE) and reliability diagrams, providing a comparison between a smaller, handcrafted model and a larger, pre-trained network.

## Project Structure
- [`CNN_calibration.ipynb`](./CNN_calibration.ipynb): Jupyter Notebook with the development of the project. Both models, LeNet-5 and DenseNet121 are studied.

## Setup

### Requirements
- Python 3.7 or higher
- Jupyter Notebook

### Usage
Open the Jupyter Notebook to view and run the code:

jupyter notebook CNN_calibration.ipynb

In the own notebook there is cell for the import of the neccessary libraries.


## Key Concepts
### 1. Reliability Diagrams
Reliability diagrams depicts the degree of calibration achieved by the model. These visualize how well the predicted probabilities match the observed frequencies. A perfectly calibrated model will have a diagonal line on the diagram.

### 2. Expected Calibration Error (ECE)
ECE quantifies the miscalibration of a model by measuring the difference between predicted probabilities and actual outcomes across multiple bins.

### 3. Temperature Scaling
Temperature Scaling is a simple yet effective post-processing technique for improving model calibration. It adjusts the logits (model outputs) by dividing them by a constant temperature parameter before applying the softmax function.


## Dataset

The dataset used in this project is the **CIFAR-10** dataset, which consists of 60,000 color images across 10 different classes. For this project, we focused on two specific classes:

- **Birds** (Class Index: 2)
- **Cats** (Class Index: 3)

The images are 32x32 pixels in RGB format. The dataset is split into a training set of 50,000 images and a test set of 10,000 images.

### Dataset Details

| **Attribute**       | **Description**                            |
|---------------------|--------------------------------------------|
| Dataset Name        | CIFAR-10                                   |
| Image Size          | 32x32 pixels                               |
| Number of Classes   | 2 (Birds, Cats)                            |
| Training Images     | 50,000                                      |
| Test Images         | 10,000                                      |

You can access the CIFAR-10 dataset from the official [CIFAR-10 website](https://www.cs.toronto.edu/~kriz/cifar.html) or download it directly using `torchvision`:

```python
from torchvision import datasets, transforms

# Download CIFAR-10
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
```

## Model Description 1: Lenet5

The core model used in this project is a **Lenet5-based Convolutional Neural Network (CNN)**, adapted specifically for binary classification tasks on the CIFAR-10 dataset. The original Lenet5 architecture was developed by **Yann LeCun** and is one of the pioneering CNN models used for image recognition.

In this project, the Lenet5 architecture has been modified to suit a binary classification problem: distinguishing between images of **cats** and **birds** from the CIFAR-10 dataset.

### Model Architecture
The adapted Lenet5 model comprises the following key layers:

1. **Convolutional Layers**:
   - Two convolutional layers with ReLU activations.
   - Feature extraction using a 5x5 kernel.

2. **Pooling Layers**:
   - Two max-pooling layers to reduce spatial dimensions and control overfitting.

3. **Fully Connected Layers**:
   - Three fully connected layers, with a final sigmoid activation for binary classification.

### Architecture Details:
| Layer Type          | Output Shape | Activation Function |
|---------------------|--------------|---------------------|
| Conv Layer 1        | (6, 28, 28)  | ReLU                |
| Max Pooling Layer 1 | (6, 14, 14)  | -                   |
| Conv Layer 2        | (16, 10, 10) | ReLU                |
| Max Pooling Layer 2 | (16, 5, 5)   | -                   |
| Fully Connected 1   | 120          | ReLU                |
| Fully Connected 2   | 84           | ReLU                |
| Output Layer        | 1            | Sigmoid             |

### Adaptations for the Project
The original Lenet5 was adapted in the following ways:

- **Dropout**: Applied to mitigate overfitting during training.
- **Batch Normalization**: Used to stabilize training and improve convergence.
- **Sigmoid Activation**: Replaced the softmax activation in the final layer, as the task requires binary classification (bird or cat).

### Performance Evaluation
The model was evaluated using the following metrics:

- **Accuracy**: The proportion of correct predictions.
- **Reliability Diagram**: A graphical representation that illustrates how well the model's predicted probabilities align with the actual outcomes.
- **Expected Calibration Error (ECE)**: A metric that measures the calibration error of a classification model.

### Training Process
The training process involves the following steps:

1. **Data Preprocessing**:
   - Filtering the CIFAR-10 dataset to include only images of birds and cats.
   - Normalizing the images to have zero mean and unit variance.

2. **Loss Function**:
   - Binary Cross-Entropy Loss.

3. **Optimizer**:
   - Adam optimizer with a learning rate of 0.001.

### Usage in the Project
The Lenet5-based model forms the backbone of the project, enabling the classification of noisy images. It was further enhanced with **Temperature Scaling** to improve calibration, resulting in better reliability diagrams and reduced ECE values.

## Results
- **Initial Model ECE**: 0.11
- **Post-Temperature Scaling ECE**: 0.023

The reliability diagram before and after applying Temperature Scaling shows significant improvement in calibration.

![Reliability Diagram. Initial vs Calibrated](/assets/images/reliability_diagram_LENET.png)

In this image, it is shown the diagramas obtained for all the different temperatura factor tried, while the diagrams for the initial (in blue) and best (in red) values are remarked. 


## Model Description 2: DenseNet-121

In addition to the Lenet5 architecture, this project employs a **DenseNet-121 model**, a deep Convolutional Neural Network (CNN) known for its efficient parameter usage and strong performance in image classification tasks. DenseNet-121 is a **pre-trained model** from the DenseNet family, which connects each layer to every other layer in a feed-forward manner. This structure allows for a more efficient flow of gradients and features throughout the network.

In this project, **DenseNet-121** was fine-tuned on a **binary classification task** using the CIFAR-10 dataset to distinguish between images of **cats** and **birds**. The model was used to investigate calibration techniques, specifically **Temperature Scaling**, to improve its reliability.

### Model Architecture
DenseNet-121 follows a unique architecture where each layer receives inputs from all preceding layers. This reduces the number of parameters and helps alleviate the vanishing gradient problem. Below is an overview of the architecture.

### Key Components of DenseNet-121:
1. **Dense Blocks**:
   - Each block consists of multiple convolutional layers.
   - Layers within a block are densely connected.

2. **Transition Layers**:
   - Used to reduce feature map dimensions between dense blocks.
   - Comprise batch normalization, ReLU activation, a convolutional layer, and average pooling.

3. **Final Classification Layer**:
   - The final layer is replaced with a binary classification head for this project.

### Architecture Details:
| Layer Type         | Output Shape | Activation Function |
|--------------------|--------------|---------------------|
| Convolutional Layer| 7x7, stride 2| ReLU                |
| Max Pooling        | 3x3          | -                   |
| Dense Block 1      | Multiple     | ReLU                |
| Transition Layer 1 | Multiple     | -                   |
| Dense Block 2      | Multiple     | ReLU                |
| Transition Layer 2 | Multiple     | -                   |
| Dense Block 3      | Multiple     | ReLU                |
| Transition Layer 3 | Multiple     | -                   |
| Dense Block 4      | Multiple     | ReLU                |
| Fully Connected Layer | 1        | Sigmoid             |

### Adaptations for the Project
The DenseNet-121 model was adapted in the following ways:

- **Pre-trained Weights**: The model was initialized with pre-trained weights from ImageNet.
- **Binary Classification Head**: The original classification head was replaced with a fully connected layer and a sigmoid activation for binary classification.
- **Fine-Tuning**: Only the final layers were retrained on the CIFAR-10 dataset to adapt the model to the new task.

### Performance Evaluation
The DenseNet-121 model was evaluated using the same metrics as the Lenet5 model:

- **Accuracy**: The proportion of correct predictions.
- **Reliability Diagram**: A graphical representation that illustrates how well the model's predicted probabilities align with the actual outcomes.
- **Expected Calibration Error (ECE)**: A metric that measures the calibration error of a classification model.

### Training Process
The training process involves the following steps:

1. **Data Augmentation and Preprocessing**:
   - Applying random rotations, crops, and horizontal flips to increase dataset variability.
   - Normalizing the images to match the pre-trained model’s input requirements.

2. **Loss Function**:
   - Binary Cross-Entropy Loss.

3. **Optimizer**:
   - Adam optimizer with a learning rate of 0.001.

### Usage in the Project
The DenseNet-121 model provides a **strong baseline for comparison** with the Lenet5 model. It highlights the calibration differences between a smaller, handcrafted network and a larger, pre-trained network. The use of **Temperature Scaling** on DenseNet-121 shows a significant improvement in calibration, as demonstrated by reduced ECE values and better reliability diagrams.

### Results
- **Initial Model ECE**: 0.093
- **Post-Temperature Scaling ECE**: 0.0043

The reliability diagram before and after applying Temperature Scaling shows a noticeable improvement in calibration.

![Reliability Diagram. Initial vs Calibrated](/assets/images/reliability_diagram_DENSE.png)

In this image, it is shown the diagrams obtained for different temperature factors applied to the DenseNet-121 model, highlighting the initial calibration (in blue) and the best-calibrated result (in red).


## Acknowledgments
This project was carried out as part of the **Machine Learning for Health Master's Program** at **Carlos III University of Madrid**.

In collaboration with:
- [Juan Muñoz Villalón](https://convertio.co/es/download/e534ebcb4fa00c329c3d9b9846013cd42fdeda/)
- [Elena Almagro](https://www.linkedin.com/in/elena-almagro-azor-a06942217/)
- [Mario Golbano](https://www.linkedin.com/in/mario-golbano-corzo/)


