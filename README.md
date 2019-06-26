# Machine Learning Developer Challenge
## Context
The Open Innovation team has developed in collaboration with a public research laboratory a new model to classify images from handwritten digits. The model, delivered in tensorflow.keras, seems to have good performances but the team needs to conduct more tests to confirm the results.
Moreover, the code is not integrated in any part of Imagia’s pipeline.

The goal of this exercise is to bring this code a step closer to industrial production by integrating it into a service.

## Starting Point
The model delivered by the research lab is a simple CNN with two convolutions. It was trained using the `tensorflow.keras` framework.

This repository contains:

* `main.py`, which is the source code delivered by the research lab
* `weights.h5`, which are the weights of the pretrained model

So far, two functionalities has been implemented:
```
python main.py --train
```

Will launch a training and overwrite weights.h5
```
python main.py --predict [path_to_image]
```

Will launch a prediction on a single image (for example `python --predict example/8.png` should output `Predicted: 8`).

## Requirements

For this challenge, you will have to integrate this code into a **very light** RESTful service which can handle the following requests:

| Request Method | Resource | Request Header | Request Body  | Response Body | Response Status Code |
| ------------- |:-------------:|:-------------:| -----:| -----:| -----:|
| TBD | http://IP:port/train | TBD | TBD | `{accuracy: x}` | TBD |
| `POST` | http://IP:port/predict | TBD | binary data to predict on. Format of file: `png` | `{prediction: x}` | TBD |

We ask you to meet the following requirements:
* Python >= 3.x
* Documentation (technical and readme to launch the service)
* We should be able to run and test the implemented solution
* Build a docker image for easier deployment
* **Optional:** if you have time / ideas, any functionality you think would add value to the service
