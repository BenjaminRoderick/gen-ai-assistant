
# Project: AI-Powered Customer Query Assistant

## Objective
Parts Avatar wants to enhance our customer support and on-site search experience using Generative AI. We want to better understand customer feedback from reviews and help customers find the right products even when they don't know the exact part name.

Your goal is to build a prototype of an AI assistant with two core capabilities: sentiment analysis and semantic product search.

## The Challenge
This project requires you to think conceptually about how to solve nuanced language problems. How do you handle a review that is both positive and negative? How do you match a vague user query to a specific product? Your choice of models, prompts, and overall strategy is more important than the complexity of the code.

## Datasets
* `data/customer_reviews.csv`: A sample of customer reviews for various products.
* `data/product_catalog.csv`: A list of products we sell.

## Your Tasks

### Part 1: Automated Sentiment Analysis
1.  **Design a Sentiment Schema:**
    * Go beyond a simple `positive/negative/neutral` classification. Design a more nuanced schema. For example, you could identify both an overall sentiment and topic-specific sentiments (e.g., `{"overall_sentiment": "mixed", "topics": {"product_quality": "positive", "shipping_experience": "negative"}}`).
2.  **Implement the Analysis:**
    * Write a Python script that takes a review and outputs its sentiment according to your schema. You can use a pre-trained model from Hugging Face, an LLM API (like OpenAI's), or any other modern NLP technique. Justify your choice.
3.  **Process the Sample Data:**
    * Run your script on the `customer_reviews.csv` and output the results.

### Part 2: Semantic Product Search
1.  **Develop a Matching Strategy:**
    * Design a system to find the best-matching products from our `product_catalog.csv` given a natural language query from a customer (e.g., "I need brake pads for a 2019 Toyota Camry that are quiet and long-lasting").
    * **Hint:** A good approach would involve creating embeddings for the product descriptions and the user query, then using vector similarity to find the closest matches.
2.  **Implement the Search Function:**
    * Write a Python function that takes a user query as input and returns the top 3 most relevant SKUs from the catalog.

### Documentation
* Update this `README.md` to be a comprehensive report of your project.
* **Crucially, explain your design choices:**
    * Describe your sentiment analysis schema and why you chose it.
    * Justify your choice of model/API for sentiment analysis.
    * Explain your semantic search strategy, including your choice of embedding model.
* Provide clear instructions on how to set up and run your code for both tasks.
* Discuss the limitations of your prototype and how you would improve it in a production environment.

## Evaluation Criteria
* **Conceptual Design & Problem-Solving:** The thoughtfulness of your sentiment schema and semantic search strategy.
* **Proficiency with Gen AI Tools:** Your choice and implementation of modern NLP models/frameworks.
* **Clarity of Explanation:** How well you justify your design decisions and explain the system's capabilities and limitations in the README.
* **Code Quality:** The clarity and organization of your implementation.

## Disclaimer: Data and Evaluation Criteria
Please be advised that the datasets utilized in this project are synthetically generated and intended for illustrative purposes only. Furthermore, they have been significantly reduced in terms of sample size and the number of features to streamline the exercise. They do not represent or correspond to any actual business data. The primary objective of this evaluation is to assess the problem-solving methodology and the strategic approach employed, not necessarily the best possible tailored solution for the data. 

## Project Report
### Sentiment Analysis
#### Sentiment Schema design
I think that a good way to approach the design of a sentiment schema for this exercise is to identify the important categories that could be discussed in a review. In my opinion, the important topics are:
- **product_quality**: This is the most important sentiment category, as it is what other customers will be looking for the most from reviews.
- **customer_service**: This is the category that many customers will evaluate to determine if they want to make a purchase.
- **product_ease_of_use**: This category refers to all issues or good experiences related to products other than the quality (ex: instructions, installation, compatibility)
- **website**: Was the website easy to navigate? Did they find the interface appealing?
- **shipping**: Shipping is outside of the control of customer support, so it deserves its own category.

#### Sentiment Analysis Pipeline
The model should perform three tasks:
1. Determine which of the above topics are discussed in the review.
2. Evaluate the sentiment on a float scale from 1 to -1.
3. Evaluate the subjectivity on a scale from 0 to 1.

I chose these three elements as the focus of my pipeline because I believe that they provide key insights for developing initiatives. Extracting the topic from a review before any other processing allows the company to understand the client's perception on key topics, this opens up opportunities to push improvements in weak areas or reinforce strengths. Secondly, evaluating sentiment on a continuous scale instead of a discrete set of sentiments, I believe that this is an important improvement because it allows the company to rank and prioritize issues that may otherwise all fall under the same label of "positive" or "negative" when, in reality, there is a significant gap in the sentiment score. Finally, I chose to include the subjectivity of a review as an output of my pipeline because it is effectively an additional indicator of sentiment that can be used to evaluate how real an issue is. For example, a very irate customer may exaggerate how bad their experience truly was in a review, whereas a more calmly worded complaint is more likely to be factual in nature. This means that, when evaluating whether to act on an issue that is experiencing a very negative customer sentiment, considering the subjectivity present in the reviews may indicate that a less drastic response is needed, which can save the company ressources.

Due to the nature of the pipeline I decided to use and my limited compute resources, instead of implementing a more traditional sentiment schema that outlines to the model all of the requirements for the classification task, I used a separate tool for each task. For the first task, I used a zero-shot classification pipeline alongside the facebook/bart-large-mnli model. I decided to do this step separately because it allows me to use a model small enough to fit in my system's memory while still retaining great performance on the specific classification task when I properly configure the input. For the second task I used a classic sentiment analysis huggingface pipeline using a roBERTa-based model. Yet again, I chose this setup to make the most of my available ressources in exchange for a small loss of performance compared to using a single LLM. For the 3rd task, I used the textblob library because of its built-in subjectivity function that is ressource-efficient.

#### Improvements For a Production Environment
In a production environment, having access to more ressources would allow me to use larger and more complex sentiment analysis models in my pipeline. The best way to take advantage of this would be to condense all of the steps of my current pipeline into a single input for the LLM. The advantages of this approach are the improved accuracy of the better model and the fact that there is no room for an error coming from a cheap model to derail the rest of the pipeline.

#### Sentiment predictions
[Here is the output of my code](https://github.com/BenjaminRoderick/gen-ai-assistant/blob/main/data/sentiment_predictions.json).

### Semantic Search
#### Matching Strategy
The strategy I decided to use for my semantic search pipeline is to create sentence embeddings for all of the product catalog, then use top-k similarity to find the items with the most similar descriptions to the sentence embedding of the user's query.

The embedding model I chose to use is Qwen/Qwen3-Embedding-0.6B. I chose this model because it is lightweight enough to be used on my hardware but produces embeddings that are very effective on downstream tasks, from my experience.

The final important decision I made regarding my semantic search pipeline is to load all of the product catalog embeddings into memory during runtime because it is the fastest and simplest method for the purposes of this task. However, the issue with this decision is the fact that this method scales very poorly as the corpus of products increases in size. Thankfully, the dataset only contains 150 entries, so the memory footprint of using this method is on the order of a few gigabytes. In a production environment, more sophisticated methods will need to be used to ensure that queries remain fast despite an immense number of records. An example of such methods would be to use a vector database due to the fact that such systems are designed with vector queries, the most critical part of this task, in mind.

### How to run the code
```
pip install -r requirements.txt
```

For sentiment analysis: [notebooks/sentiment_analysis.ipynb](https://github.com/BenjaminRoderick/gen-ai-assistant/blob/main/notebooks/sentiment_analysis.ipynb)

For semantic search: [notebooks/semantic_search.ipynb](https://github.com/BenjaminRoderick/gen-ai-assistant/blob/main/notebooks/semantic_search.ipynb)

Both notebooks are designed such that they can be run sequentially to prepare the data and run the corresponding pipeline.
