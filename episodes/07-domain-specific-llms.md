---
title: '7. Domain-Specific LLMs'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

•	How can tune the LLMs to be domain-specific? 
•	What are some available approaches to empower LLMs solve specific research problems? 
•	Which approach should I use for my research? 
•	What are the challenges and trade-offs of domain-specific LLMs?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

•	Be able to identify approaches by which LLMs can be tuned for solving research problems.
•	Be able to use introductory approaches for creating domain-specific LLMs.

::::::::::::::::::::::::::::::::::::::::::::::::

### 7.1.	Introduction to DSL (Available Approaches)

To enhance the response quality of an LLM for solving specific problems we need to use strategies by which we can tune the LLM. Generally, there are four ways to enhance the performance of LLMs:

#### 1. Prompt Optimization:
To elicit specific and accurate responses from LLMs by designing prompts strategically. Zero-shot Prompting: This is the simplest form of prompting where the LLM is given a task or question without any context or examples. It relies on the LLM’s pre-existing knowledge to generate a response. Example: “What is the capital of France?” The LLM would respond with “Paris” based on its internal knowledge. Few-shot Prompting: In this technique, the LLM is provided with a few examples to demonstrate the expected response format or content. Example: To determine sentiment, you might provide examples like “I love sunny days. (+1)” and “I hate traffic. (-1)” before asking the LLM to analyze a new sentence.

#### 2. Retrieval Augmented Generation (RAG):
To supplement the LLM’s generative capabilities with information retrieved from external databases or documents. Retrieval: The LLM queries a database to find relevant information that can inform its response. Example: If asked about recent scientific discoveries, the LLM might retrieve articles or papers on the topic. Generation: After retrieving the information, the LLM integrates it into a coherent response. Example: Using the retrieved scientific articles, the LLM could generate a summary of the latest findings in a particular field.

#### 3. Fine-Tuning: 
To adapt a general-purpose LLM to excel at a specific task or within a particular domain. Language Modeling Task Fine-Tuning: This involves training the LLM on a large corpus of text to improve its ability to predict the next word or phrase in a sentence. Example: An LLM fine-tuned on legal documents would become better at generating text that resembles legal writing. Supervised Q&A Fine-Tuning: Here, the LLM is trained on a dataset of question-answer pairs to enhance its performance on Q&A tasks.
Example: An LLM fine-tuned with medical Q&A pairs would provide more accurate responses to health-related inquiries.

#### 4.	Training from Scratch: 
Builds a model specifically for a domain, using relevant data from the ground up.


::::::::::::::::::::::::::::::::::::: Discussion

## Discuss in groups. 

Which approach do you think is more computation-intensive? Which is more accurate? How are these qualities related?  Evaluate the trade-offs between fine-tuning and other approaches.

![image](https://github.com/sabah-gaznaghi/intro-nlp-llm/assets/45458783/7526be09-de4d-4c59-b6df-93f0d2dca9d7)

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: Discussion

## Discuss in groups. 

What is DSL and why are they useful for research tasks? Think of some examples of NLP tasks that require domain-specific LLMs, such as literature review, patent analysis, or material discovery. How do domain-specific LLMs improve the performance and accuracy of these tasks?

![image](https://github.com/sabah-gaznaghi/intro-nlp-llm/assets/45458783/6e5a6a74-eaa2-430d-b5b5-8198716fb4f7)




A: Stemming
B: Word2Vec	
C: Text Preprocessing	
D: Part-of-Speech Tagging	
E: Stop-words Removal	
F: Transformers	
G: Bag of Words	
H: Tokenization	
I: BERT	
J: Lemmatization


```
1.	“A statistical approach to modeling the meaning of words based on their context.”
[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J

2.	“A process of reducing words to their root form, enabling the analysis of word frequency.”
[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J

3.	“An algorithm that uses neural networks to understand the relationships and meanings in human language.”
[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J

4.	“A technique for identifying the parts of speech for each word in a given sentence.”
[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J

5.	“A method for cleaning and preparing text data before analysis.”
[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J

6.	“A library that provides tools for machine learning and statistical modeling.”
[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J

7.	“A model that predicts the next word in a sentence based on the words that come before it.”
[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J

8.	“A framework for building and training neural networks to understand and generate human language.”
[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J

9.	“A technique that groups similar words together in vector space.”
[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J

10.	“A method for removing commonly used words that carry little meaning.”
[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J

```

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: Discussion

## Discuss in groups. Share insights on how NLP can be applied in your field of interest.


:::::::::::::::::::::::: solution 

#### Field of Interest: Environmental Science
•	NLP for Climate Change Research: How can NLP help in analyzing large volumes of research papers on climate change to identify trends and gaps in the literature?
•	Social Media Analysis for Environmental Campaigns: Discuss the use of sentiment analysis to gauge public opinion on environmental policies.
•	Automating Environmental Compliance: Share insights on how NLP can streamline the process of checking compliance with environmental regulations in corporate documents.

#### Field of Interest: Education
•	Personalized Learning: Explore the potential of NLP in creating personalized learning experiences by analyzing student feedback and performance.
•	Content Summarization: Discuss the benefits of using NLP to summarize educational content for quick revision.
•	Language Learning: Share thoughts on the role of NLP in developing language learning applications that adapt to the learner’s proficiency level.

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::



::::::::::::::::::::::::::::::::::::: challenge

## Challenge Using an LLM (Mini-Project)
Context Example: Environmental science and climate change
Using Hugging Face model distilbert-base-uncased and Few-Shot Prompting: To improve the model’s performance in answering field-specific questions, we will use few-shot prompting by providing examples of questions and answers related to environmental topics.

```python
from transformers import pipeline
# Initialize the question-answering pipeline with DistilBERT
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased')

# Few-shot prompting with examples
context = """
Question: What is the greenhouse effect?
Answer: The greenhouse effect is a natural process that warms the Earth's surface.

Question: How can we reduce carbon emissions?
Answer: We can reduce carbon emissions by using renewable energy sources, improving energy efficiency, and planting trees.

Question: What are the consequences of deforestation?
Answer: Deforestation can lead to loss of biodiversity, increased greenhouse gas emissions, and disruption of water cycles.
"""

# User's field-specific question
user_question = "What can individuals do to combat climate change?"

# Prepare the prompt for the model
prompt = {
    'context': context,
    'question': user_question
}

# Get the answer from the model
response = qa_pipeline(prompt)
print(response['answer'])
```
:::::::::::::::::::::::: solution 

The model should provide a relevant answer based on the few-shot examples provided. For instance, it might say: “Individuals can combat climate change by reducing their carbon footprint, using less energy, recycling, and supporting eco-friendly policies.” In this challenge, we used the **distilbert-base-uncased** model from Hugging Face’s transformers library to create a question-answering system. Few-shot prompting was employed to give the model context about environmental topics, which helps it generate more accurate answers to user queries. The **qa_pipeline** function is used to pass the prompt to the model, which then processes the information and returns an answer to the user’s question. This mini-project showcases how LLMs can be fine-tuned to specific fields of interest, providing valuable assistance in answering domain-specific queries.

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

### 8.2.	Further Resources

For continued learning, here are detailed resources:
•	Natural Language Processing Specialization (Coursera): A series of courses that cover NLP foundations, algorithms, and how to build NLP applications.
•	Stanford NLP Group: Access to pioneering NLP research, datasets, and tools like Stanford Parser and Stanford POS Tagger.
•	Hugging Face: A platform for sharing and collaborating on ML models, with a focus on democratizing NLP technologies.
•	Kaggle: An online community for data scientists, offering datasets, notebooks, and competitions to practice and improve your NLP skills.
Each resource is a gateway to further knowledge, community engagement, and hands-on experience.


### 8.3.	Feedback

Please help us improve by answering the following survey questions:
1.	How would you rate the overall quality of the workshop?
   
[ ] Excellent,  [ ] Good,  [ ] Average,  [ ] Below Average,  [ ] Poor


2.	Was the pace of the workshop appropriate?
   
[ ] Too fast,     [ ] Just right,     [ ] Too slow


3.	How clear were the instructions and explanations?
   
[ ] Very clear,     [ ] Clear,     [ ] Somewhat clear,      [ ] Not clear


4.	What was the most valuable part of the workshop for you?
   

5.	How can we improve the workshop for future participants?


Your feedback is crucial for us to evolve and enhance the learning experience.


::::::::::::::::::::::::::::::::::::: keypoints 

•	Various NLP techniques from preprocessing to advanced LLMs are reviewed.
•	NLPs’ transformative potential provides real-world applications in diverse fields.
•	Few-shot learning can enhance the performance of LLMs for specific field of research. 
•	Valuable resources are highlighted for continued learning and exploration in the field of NLP.

::::::::::::::::::::::::::::::::::::::::::::::::

<!-- Collect your link references at the bottom of your document -->to other hyperparameters, the choice of optimizer depends on the problem you are trying to solve, your model architecture, and your data. Adam is a good starting point though, which is why we chose it. Adam has a number of parameters, but the default values work well for most problems so we will use it with its default parameters.
