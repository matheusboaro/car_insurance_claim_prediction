# Car Insurance Prediction Using TabNet

## Proposed Methodology

![alt text](https://github.com/matheusboaro/car_insurance_claim_prediction/blob/main/methodology.png)

## TabNet Overview

Developed by Google, the TabNet model is based on the idea of self-attachment introduced by transformer models for problem solving in NLP. According to the tests performed, this architecture was able to outperform the metrics obtained by other approaches in several benchmarks. 

The main advantage of using transformer-based depleaning models, besides its generalization power, is the interpretative capacity of the results, something that is currently gaining more and more importance, since understanding how intelligence algorithms make decisions can help us to clearly understand user behavior, for example. Thus, TabNet was selected for being able to give us the power of non-linear problem solving coming from deep neural network models, with the explainability that more classical models such as decision trees can give us.

## Tasks Adressed

### Claim Flag Prediction

In this task, the main goal is to identify whether or not a customer will open a claim notice. Within the database used, this action is indicated by the attribute "CLAIM_FLAG". In order not to bias the data, the attribute "CLM_AMT", which indicates the amount requested by the insured, was also removed, since it is completely related to whether the customer has filed a claim or not.

### Claim Amount Prediction

This task consisted of giving a customer who has filed a claim notice, what amount he will claim. In the database used this value is indicated by the field "CLM_AMT". For this step, the data was also normalized between 0 and 1.

## Results
### Claim Flag Prediction

- TabNet - 0.74 (F1-Score)
- XGBoost - 0.75 (F1-Score)

### Claim Amount Prediction

- TabNet - 1952.09 (mae)
- XGBoost - 1958.45 (mae)

## TabNet Explanability 

![alt text](https://github.com/matheusboaro/car_insurance_claim_prediction/blob/main/tabnet_masks.png)


## Conclusions

### What drives claims ?

According to the study conducted and presented in this notebook, it is not just one customer attribute that tells if he will submit a claim or not, but several attributes, the main ones being his salary, his history of previous claims, his location, and the purpose for which he uses his car. Thus, the proposed solution can explain what leads a driver to submit a claim in a general way, but we can also visualize in a specific way, which characteristics most influenced the model to classify that user (attention mask visualizations).

### Can we predict the value of a claim based on driver's profile?

As shown, we are also able to assess the value on which the customer will request based on his profile. According to the model, the requested value is related to attributes such as gender, education level, marital status, car usage, and whether the customer has had his or her driver's license renewed or not.

### What are the advatanges of the proposed aproach ?

Competitive results in comparision with similiar works and good explainability, plus as demonstrated by the authors of the TabNet architecture, this approach is able to address the actual importance of the attributes relative to the target, algorithms such as XGBoost, have been shown to be more likely to better balance the importance of the attributes.

### What are the disadvantages of the proposed aproach ?

Neural nets, such as TabNet, tend to have a higher algorithmic complexity, which leads to a higher computational cost compared to training other approaches, such as decision trees. This problem can be alleviated through the use of GPU's, parallelizing some processing steps in order to decrease training time.

## Next Steps

### Unsupervised Learning

With this it is hoped that the model can find the non-linear relationships between the data without the need for labels. Studies show that initializing the TabNet weights in an unsupervised manner decreases the model optimization time to reach an optimal result in a more efficient way, besides becoming more robust to unseen data.

### HyperParameter optimization

Perform an optimization step in order to find the hyperparameters that best fit the proposed problem, thus maximizing the model results

### Go Deep

Test other deep learning models for tabular data to find more insights about the data and customers.

### Hybrid Model

Build a hybrid model (ensemble) that can take advantage of the main positive points of each approach and mitigate the negative points.


## Observations

The dataset used is private, but you can find a close related dataset [here](https://www.kaggle.com/datasets/xiaomengsun/car-insurance-claim-data).

## References

https://github.com/xzhangfox/Prediction-of-Car-Insurance-Claims/blob/master/Final-Group-Project-Report/FinalReport.pdf

https://www.kaggle.com/code/dllim1/end-to-end-ml-on-motor-insurance-with-xgboost/comments

https://github.com/dreamquark-ai/tabnet

https://arxiv.org/pdf/1908.07442.pdf
