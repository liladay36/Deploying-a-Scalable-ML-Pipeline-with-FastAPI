# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Model Name**: Income Classification Model
- **Model Type**: Random Forest Classifier
- **Version**: 1.0

## Intended Use
 **Use Case**: This model is intended to predict whether individuals make more that $50,000 annualy based on various attributes like demographics, and employment
- **Target Audience**: Data scientists, machine learning engineers, and researchers developing fairness-aware ML pipelines or studying model performance across data slices.

## Training Data
- **Dataset**: The model was trained on the Census Income dataset .
- **Size**: Number of samples in the training dataset was 32561 entries, 26000 after split
- **Features**: The features in the data set included workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Data Source**: [Where the data was obtained, e.g., Kaggle, public datasets]

## Evaluation Data
 **Dataset**: The model was evaluated on a hold-out validation set (20% of the original dataset)
- **Size**: 6,500 records
- **Features**: Same as training dataset.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
Precision: 0.7481 - Good: correct predictions for earners >$50K, is correct about 75% of the time.
Recall: 0.6334: Poor - catches 63% of all people who actually make >$50K. But false negatives are missed 1/3 of the time
F1: 0.6860 - Okay performance for real world tabular data, but not necesarilly conclusive. 

## Ethical Considerations
- **Bias**: The model was tested based on historical data. This can have implications of racial disparities, sosietal impact and can influence output. 
- **Fairness**: Performance can vary when tested against different demographics of people. However, the performance_on_categorical_slice function was used to evaluate model fairness, and results per feature group (e.g., by race, sex) are recorded for transparency.

## Caveats and Recommendations
- **Limitations**: It may not do well for populations or contexts outside the U.S. Census demographics or even state by state where there a large cultural differences. 
- **Recommendations**: Retrain the model consistently as more data and demographics arise and maintain relevance. 