# Live CCTV Flood Detection with ResNet
## About the project
This project aims to solve the challenge set by the Met Office's Flood Forecasting Centre at the Hatch:IT Competition: Using existing distributed networks to aid real-time flood detection and damage assessment.
We propose utilising existing CCTV cameras in place, which can be available publicly, privately or by partnering with enterprises. Subsequently, therefore we developed and trained a computer vision model to be able to detect collecting water on road surfaces in real time from a constant feed of CCTV footage. The CCTV in use for this prototype is live from the Abbey Road crossing in London and a demo output from the system can be seen as follows:

<br>
</br>

![Demo](demo/demo_output.png)

## Training data
Training data was obtained from the following Kaggle datasets

[BDD100K](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k)

[FloodIMG](https://www.kaggle.com/datasets/hhrclemson/flooding-image-dataset)

## Model Performance
The trained ResNet model achieved 99.25% accuracy on validation data, although more quality and specific training data would definitely help in more accurate CCTV stream performance.
