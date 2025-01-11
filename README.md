# CNNs_calibration
# Calibration in CNNs for Bid-Cat classification

## Project Description
This project explores the calibration of Convolutional Neural Networks (CNNs) for binary classification, specifically focusing on improving model reliability through Temperature Scaling. The project uses the CIFAR-10 dataset, training a **LeNet-5 CNN** to distinguish between birds and cats, and evaluates calibration metrics like Expected Calibration Error (ECE) and reliability diagrams.

## Contents
- **Notebook**: Python code for model training and calibration.
- **Figures**: Visual representations of reliability diagrams and temperature scaling effects.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/calibration-cnn.git
   ```
2. Navigate to the project folder:
   ```bash
   cd calibration-cnn
   ```
3. Run the notebook in Jupyter:
   ```bash
   jupyter notebook Project_II_DL.ipynb
   ```

## Key Concepts
### 1. Reliability Diagrams
Reliability diagrams depicts the degree of calibration achieved by the model. These visualize how well the predicted probabilities match the observed frequencies. A perfectly calibrated model will have a diagonal line on the diagram.

### 2. Expected Calibration Error (ECE)
ECE quantifies the miscalibration of a model by measuring the difference between predicted probabilities and actual outcomes across multiple bins.

### 3. Temperature Scaling
Temperature Scaling is a simple yet effective post-processing technique for improving model calibration. It adjusts the logits (model outputs) by dividing them by a constant temperature parameter before applying the softmax function.

## Results
- **Initial Model ECE**: 0.047
- **Post-Temperature Scaling ECE**: 0.021

The reliability diagram before and after applying Temperature Scaling shows significant improvement in calibration.

## Acknowledgments
This project was carried out as part of the **Machine Learning for Health Master's Program** at **Carlos III University of Madrid**.

In collaboration with:
- [Juan Muñoz Villalón](https://convertio.co/es/download/e534ebcb4fa00c329c3d9b9846013cd42fdeda/)
- [Elena Almagro](https://www.linkedin.com/in/elena-almagro-azor-a06942217/)
- [Mario Golbano](https://www.linkedin.com/in/mario-golbano-corzo/)

## Full Report
Download the complete project report:
[Download Full Report PDF](/assets/pdf/Project_II_DL.pdf)

