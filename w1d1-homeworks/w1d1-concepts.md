### Answer exercise A : Classify Real-World ML Problems
| Scenario | Problem Type | Input Features | Output |
|----------|--------------|----------------|--------|
| Preditct whether a loan application will detault | Classification | Credit Scoring, Income, Employment Status | Default / Not Detault) |
| Forecast next month's energy consumption for a city | Regressio | Historical energy usage, weather data, population/activity trens | Numeric value|
| Group customers by purchasing behavior | Clustering | Purchase history,frequency of purchases,average spend | Customer segments/groups|
| Detect fraudulent credit card transactions | Classification | Transaction amount,location,time | Fraud/Not Fraud) |
| Generate product descriptions from an image | Generative | Image (pixels/features) | Text desctiption of the product |
| Predict the severity (1-5) of a patient's condition | Classification (ordinal) | Symptoms, vital signs, lab results | Severity category(1-5) |

------
### Answser exercise B : Map the ML Lifecycle
**Scenario:** A hospital wants to predict which patients admintted to the ER are at high risk of readmission within 30 days.
```
1. Problem Definition : Predict if an ER patient will return within 30 days. It is a binary classification problem (high risk or low risk). The goal is to reduce readmissions.
2. Data Collection : Collect patient data like age, history, diagnosis, and past visits. Include lab results and treatments. Use past data with known outcomes.
3. EDA & Preprocessing : Check the data for patterns and missing values. Clean the data and convert categories into numbers. Remove unnecessary features.
4. Model training : Train a model using the data (e.g., logistic regression). Split data into training and testing sets. The model learns to predict risk.
5. Evaluation : Measure performance using accuracy, precision, and recall. Focus on correctly identifying high-risk patients. Make sure the model works well on new data.
6. Deployment : Use the model in the hospital system. It predicts risk when a patient arrives. Doctors can use it to make decisions.
7. Track model performance over time. Check if accuracy drops. Retrain the model if needed.
```


### Answer exercise C : AI vs ML vs Deep Learning Sorting
| System | Type | Reason|
|--------|------|-------|
