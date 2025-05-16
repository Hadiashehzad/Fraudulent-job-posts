# Fraudulent-job-posts
Spotting Fake Job Posts

# Introduction

Although the online job market is filled with opportunities, not all listed jobs are legitimate. As is often the case, job postings are sometimes created to scam job seekers. The scammers trick them into revealing their sensitive personal information, which can include their address, social security number, or credit card details. Other malicious intent also includes attracting job seekers with a lucrative job opportunity but then asking for payment or processing fees in return.  Linkedin has become a profitable hub for online job scammers. 

Our project aims to identify fake job posts on the internet using machine learning models. We used a labelled job postings dataset from Kaggle that contains 18000 job listings, and used it to train our models. The models used were Stochastic Gradient Descent and Long Short Term Memory (LSTM) Sequential Model. 

Our project follows the standard data science lifecycle, which is as follows:
1.	Problem Formulation
2.	Acquire and Clean the Data
3.	Exploratory Data Analysis
4.	Run Predictions and Inference
5.	Evaluate the model and draw conclusions


# Methodology

The training dataset we used for the job listings is from Kaggle, and can be found here. It contains several text and numerical features of a job posting, like title, description, location, and salary range, and the target variable on whether the job is fraudulent or not. The total number of jobs are 18000, out of which around 800 are fraudulent. This amounts to only 4.44% of jobs being fraudulent, which is not surprising, as the number of fake jobs as compared to real jobs on the internet are a few.

We combined the text features into a single string of text to allow for easy modeling. For numeric features, we retained three important features out of all that were provided. We ran the following models on the dataset:
1.	Random Forest Classifier
2.	Stochastic Gradient Descent
# Problem Formulation
Our project’s aim is to create a classifier model that can predict whether a job on the internet is fraudulent or not, when given its features. The features include both text and numeric data, so we have processed them both separately to be used in the models. 

Data Cleaning
The dataset contained 17880 job postings with 18 features. The features are defined below:

| Serial # | Feature Name |	Data type |	Description |
| -------- | ------------ | --------- | ----------- |
| 1 |	job_id | int64 | Id number of each job posting |
|2|	title|	object|	Title of the job posting|
|3|	location|	object|	Where the job is location|
|4|	department|	object|	Which functional department the job is in|
|5|	salary_range|	object|	The salary range of the job|
|6|	company_profile|	object|	A description of the company the job is in|
|7	|description|	object|	A brief description about the job |
|8	|requirements|	object|	Prerequisites to be eligible for the job|
|9	|benefits|	object|	Additional benefits and perks of the job|
|10	|telecommuting|	int64|	Whether the job is online or in-person|
|11	|has_company_logo|	int64|	Whether the job has a company logo|
|12	|has_questions|	int64	|Whether the job has questions|
|13	|employment_type|	object|	Full-time, part-time, contract, temporary and other|
|14	|required_experience|	object	|Experience required to qualify for the job|
|15	|required_education|	object|	Education required to qualify for the job|
|16	|industry|	object|	Which industry the job is relevant to|
|17	|function|	object|	What is the functionality of the job|
|18|	fraudulent|	int64	|The target variable: 0 for real, 1 for fake|

A thorough cleaning phase of the dataset became essential before moving forward because accurate results required it. The analysis started by addressing absent data in both job groups as well as job descriptions. Many variables like “department” and “salary_range” had a lot of missing values so they had to be excluded from the dataset to gain meaningful results.
The presence of special characters caused issues mainly within the company profile, description, requirements, and benefits sections. The fields included useless symbols that were eliminated without deleting essential punctuation. The maintenance of readability combined with textual data consistency improved via this process.

# Exploratory Data Analysis

As noted earlier, only a small proportion of the jobs were fraudulent. After data cleaning, we were left with 17880 jobs out of which 865 were fraudulent, amounting to 4.6% of the entire dataset.

![image](https://github.com/user-attachments/assets/2f3db298-ae48-4d88-ae0c-4c17268fa59b)

Many of the variables are of text type so we did not calculate any descriptive stats. We analysed the correlation matrix between all numeric features to check for multicollinearity.

 ![image](https://github.com/user-attachments/assets/c36c5b68-5e0c-4159-9bdf-9c00ca060bd8)
 
The correlation matrix does not exhibit any strong positive or negative correlations between the numeric data.

![image](https://github.com/user-attachments/assets/97914760-4b30-401d-ba32-321919d8b2ee)

An examination was conducted which evaluated how different employment types impacted the probability of receiving a fraudulent position.
The bar plot demonstrates that full-time jobs constitute majorities in fraudulent and authentic vacancies. The percentage of deceptive postings is relatively higher in the full-time employment category than in the part-time or contract or temporary categories.
Full-time position attractiveness serves as a key tool for scammers who want to deceive potential job candidates.

![image](https://github.com/user-attachments/assets/b6b1456b-0ae5-4ea0-a390-e9c5fb0e4bd0)
 
An examination was conducted which evaluated how different employment types impacted the probability of receiving a fraudulent position.
The bar plot demonstrates that full-time jobs constitute majorities in fraudulent and authentic vacancies. The percentage of deceptive postings is relatively higher in the full-time employment category than in the part-time or contract or temporary categories.
Full-time position attractiveness serves as a key tool for scammers who want to deceive potential job candidates.

![image](https://github.com/user-attachments/assets/ae48ffe2-f89f-4fdb-8fb7-abb09ae19499)
 
The study investigated the relationship between jobs available for remote work and fraudulent job posts.
Remote employment positions with Telecommuting=True show a minor increase in suspicious listings relative to regular workplace jobs with Telecommuting=False. The lower volume of remote jobs does not deter scammers from advertising them because remote jobs allow them to reach a larger audience without physical office requirements.

![image](https://github.com/user-attachments/assets/9a9c1d07-99ba-4405-b474-f3f7b6000d2f)
 
We explored a histogram describing a character count across all text features to visualize the difference between real and fraudulent jobs. We can see that even though the character count is fairly similar for both real and fake jobs, real jobs have a higher frequency.
# Models
Having explored the dataset, we were at the stage to finalise our features for the classification models. We combined all the text features into a single string of text to allow for easier tokenisation and lemmatization later on. The features included in the text feature were:

1. title 
2. location 
3. company_profile 
4. description 
5. requirements 
6. required_experience	
7. required_education
8. benefits
9. industry
10. function 

We had to now process our text feature so it could be fed into our models. Below is a breakdown of the text processing steps we carried out:

1.	**Tokenisation:**
Raw text can't be processed directly so we need individual units (tokens) like words or subwords. This helps the model to analyze meaning word-by-word.
2.	**Lowercasing:**
Converts all text to lowercase letters to avoid duplicate meanings caused by capitalization differences.
3.	**Stopword Removal:**
Removes common, low-information words (like “the”, “is’”, “and”) that don't help the model distinguish between texts.
4.	**Lemmatization:**
Maps words to their root or base form (e.g., "was" → "be", "dogs" → "dog") to normalize different word forms. This helps the model understand concepts rather than getting distracted by different forms of the same word.

The final features left in our dataset were:
1.	**Telecommuting**
2.	**Character_count**
3.	**Text**

# Stochastic Gradient Descent

**Approach**

We used the Stochastic Gradient Descent (SGD) model to classify job postings as fraudulent or not. Our SGD classifier trains a linear logistic regression model to minimise the log-loss of the predictions. In general, SGD trains a model by updating parameters step-by-step using small batches instead of the full dataset at once. We represented the text feature as a bag of words counts, creating a sparse matrix suitable for SGD. SGD handles sparse matrices very efficiently because it updates weights without needing to load the full data into memory.

**Training**

We trained two separate SGD classifiers on the entire dataset, one using only the text feature, and the other using only the numeric features. This would make it easier for the model to handle text and numeric features separately. As we had count vectorised or text feature, it got transformed into a huge and sparse matrix (one column for each word and mostly filled with zeros because not every word appears in every job post). The numeric features were few but they carried strong structured information, like the length of the job description or the telecommuting status. Using the features together to train a single model would overwhelm the few numeric features, and the classifier would mainly focus on words because there are so many word features. Training two separate models would preserve the strength of both types separately and let them specialise in text and numeric patterns.

**Results**

Below are the evaluation metrics for both the models’ performance:

|Metric	|Text model|	Numeric Model|
|------|-----------|--------------|
|Accuracy|	0.987|	0.966|
|Precision|	0.896|	0.0|
|Recall	|0.715|	0.0|
|F1 score|	0.795|	0.0|


We can clearly see that the numeric model has failed to make any correct predictions as its precision and recall both are zero. An accuracy of 0.966 is simply because of the high class imbalance, where only 4.8% jobs were fraudulent. Even after predicting all false negatives, it would still achieve a high accuracy because of the class imbalance.

For the text model, the results on all metrics are quite promising. Its accuracy is 0.987 which is high but again can be owed to the class imbalance. A precision of 0.869 and recall of 0.715 tells us that the model is around 90% correct in predicting fraudulent jobs and is catching around 71% of all frauds. The f1 score of 0.795 shows a very healthy balance between catching frauds and avoiding false alarms.

One potential reason for a poor recall could be that the Count Vectorizer approach is too simple in capturing the meaning of the text tokens. It only stores the raw counts of the words in each job post, and it doesn’t understand the importance or semantics of the words across documents the way more advanced text embeddings would do. This made us try a more complex model to capture the semantic meanings of the text and use it to classify the jobs

# Sequential Model

**Approach**

The chosen methodology involves the utilization of Sequence Models, with a specific focus on Recurrent Neural Networks (RNNs) augmented with embeddings. We utilized the GPT-4All library to generate embeddings of the text feature, providing a contextualized representation of text features of the jobs that manages to capture semantic meaning and context.

**Model Architecture**

The selected model architecture is based around the use of Long Short-Term Memory (LSTM) networks. LSTMs are employed due to their ability to capture long-range dependencies in sequential data. The model structure comprises an LSTM layer followed by a fully connected layer designed for binary classification. 

**Training Methodology**

The training process involves the use of binary cross-entropy loss and the Adam optimizer. The training loop is executed over 60 epochs, allowing the model to learn the intricacies of the training data. The model's performance is continuously monitored through the tracking of both loss and accuracy metrics. 

**Results**

The training accuracy exhibits steady improvement over epochs, reaching an impressive value of 100% after 60 epochs. The model undergoes thorough assessment on a separate validation set, yielding an accuracy of 98.33%. 


|Metric|	Score|
|------|------|
|Accuracy	|0.983|
|Precision|	0.851|
|Recall|	0.622|
|F1 score	|0.719|


The high - exactly perfect -  training accuracy suggests that the model has learned the training data well. However, the lower validation accuracy indicates a potential overfitting issue or a need for further model tuning. Experimentation with hyperparameters and model architecture may be beneficial to improve generalization to unseen data. 

The precision is high, indicating that the model is mostly correct in predicting the fraudulent jobs, however, the recall is low. This suggests that our use of a more complex approach - the use of the GPT4all library - to generate text embeddings for the text data did not result in better identification of fraudulent jobs than the simpler SGD model that only used the frequency of words. The embeddings might be missing out on important signals in the text that are uncommon occurrences.


# Evaluation
The reasons behind evaluation scores being low can be because of: 
1.	Class Imbalance: The dataset had a higher number of real jobs (17,880) than fraudulent (865), which might be causing the models to get biased towards real jobs. We can also see from the evaluation metrics that all the three models (including the SGD on numeric data) had high accuracy but a lower recall, which could be because of the class imbalance. 
2.	Low Recall: All the models had a lower recall which means that they were not highly successful in predicting the true fraudulent jobs. With the primary reason being a high class imbalance, other reasons could include feature overlap between the jobs. As recognised during our EDA phase, many fraudulent jobs showed patterns very similar to real jobs, for example in the character count in text features of the job or whether the company has a logo or not.
3.	Bias: The dataset was marked manually by human annotators. Hence, it does contain the bias in it. The job postings being marked real or fraudulent are actually in the human annotators’ perspective which might not be reflected in some model’s embeddings’ perspective. 

# Best Model

The best model out of the three was the Stochastic Gradient Descent model trained only on the text features. It could be because of the nature of the dataset and the problem at hand. LSTMs are suitable for capturing long-term dependencies and sequential patterns, such as in time-series data or long and complicated text data. However, in this case, the text data is not sequentially dependent in a way that requires deep contextual understanding — a simple count-based feature representation is sufficient for identifying the fraudulent behavior. SGD with logistic loss is more efficient for this task because it directly operates on high-dimensional sparse data from the count vectorizer, making it faster and more accurate for our classification task.

Furthermore, the text SGD model on the text feature is outperforming the numeric feature model due to the richness of the textual feature that has been created by combining ten text features of the dataset. The text feature contains a lot of nuanced information, such as the job title, description, location, and the company profile, which are highly indicative of fraudulent behavior. The count vectorizer extracts these important features from the text, which the SGD model leverages to identify patterns effectively. On the other hand, numeric features like telecommuting and character_count are less expressive and do not carry as much discriminative power for the classification task. 

|Metric|	Text SGD model	|Numeric SGD Model|	LSTM|
| ---| --- | ---- | --- |
|Accuracy|	0.987|	0.966|	0.983|
|Precision|	0.896	|0.0|	0.851|
|Recall	|0.715	|0.0	|0.622|
|F1 score|	0.795|	0.0|	0.719|

As we can see from the comparison above, the SGD model with text features performed significantly well even with the huge class imbalance in the dataset.
