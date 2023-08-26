# CosDefense
This is the reference code for implementing our CosDefense mechanism proposed in [Secure Federated Learning against Model Poisoning Attacks via Client Filtering](https://arxiv.org/pdf/2304.00160.pdf). 


## Setup Environment

Required packages can be installed by running the following command:
```
pip install -r requirements.txt
```
## Code Structure
```main.py``` contains main function to run experiments.\
```models.py``` contains model definitions.\
```utils.py``` contains helper functions and defense algorithms including Krum, Clipping-Median, and CosDefense.

## Experiments
You can change the parameters (including dataset, attack type and defense mechanism) in the beginning of the ```main.py``` file to do experiments with different settings and run the experiment by:
```
python3 main.py
```

## References
Our code structure is built on top of the https://github.com/SliencerX/Learning-to-Attack-Federated-Learning
