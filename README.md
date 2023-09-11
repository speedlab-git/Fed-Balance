# FedAVO





## Installation

First, you need to install the project dependencies. To do this, navigate to the project directory and run the following command:

```
pip install -r requirements.txt
```

This will install all the necessary packages for the project.

## Running the Project

To run the project, use the following command in your terminal:

**Example**: 
```
python3 main.py --dataset=ucf \
--data_split=iid --num_rounds=400 \
--clients_per_round=10 --batch_size = 4\
--num_epochs=5 --tuning_epoch=3 --train_samples=500
```

This will execute the `main.py` file, which contains the main code for the project. 





## EXPERIMENTAL RESULTS

<img src="figures/FashionAcccorr.png" width="250"> <img src="figures/non-iid-CIFAR.png" width="250"> <img src="figures/non-iid-mnist_upd.png" width="250">

## REFERENCES 
See our [FedAVO](https://arxiv.org/abs/2305.01154) paper for more details as well as all references.


