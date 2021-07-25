## Class balance
At the moment the dataset contains almost double the instances of people that earn less than 50,000 per year compared to the ones with more than 50,000 per year.

For this exercise, the larger data class is subsampled to half its size and then accuracy is used for evaluation.

The next step is to experiment with increasing this class size and find a trade-off between representing a real world data distribution and overfitting. The task, e.g., classification or prediction, and the associated risks with false positives and false negatives should be considered.

## Evaluation
In this exercise the data is split naively with the 80-20 rule, holding a random 20% of the data for predictions in a stratified manner.
This setting is suboptimal and the next step would be to either repeat this evaluation for k iterations, or perform cross validation to have more reliable results and avoid overfitting.

## Data representation
The embedding size is calculated with a rule of thumb (embedding size = min(50, number of categories/2)). This is experimental, thus alternative values can be used, e.g., 50 or 100.

## Model
A simple dense network is used with 100 neurons and a rectifier activation function. The next step is to try more complex architectures like a CNN and pooling layers before the rectifier to reduce overfitting, especially if the whole imbalanced dataset is considered. 
The training parameters for the number of epochs and batch size is set to 20 and 32 respectively, and they can be tuned automatically, for instance with talos library for hyperparameter optimization.