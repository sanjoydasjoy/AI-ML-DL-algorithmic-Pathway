# k-nearest neighbors (KNN):

The k-nearest neighbors (KNN) algorithm is a non-parametric, supervised learning classifier that uses proximity to make classifications or predictions about the grouping of an individual data point. It is one of the most popular and simplest classification and regression algorithms used in machine learning today.


## Proximity and Closeness
The KNN algorithm relies on the concept of proximity or closeness to classify data points. For instance, if you have a dataset of **fruits** with features such as **weight** and **sweetness**, and you want to classify a **new fruit**, KNN will measure the **distance** between this new fruit and all other fruits in the dataset. For K = 3, if the **among the closest three fruits** are **two apples** and **one orange**, the algorithm will classify the **new fruit** as **an apple** due to its **proximity** to those closest examples.

## Non-Parametric Nature
KNN is considered non-parametric because it makes no assumptions about the underlying data distribution (**data distributions**: Remember in statistics, we used frequency distribution table) . For example, in a dataset of animal species based on features like weight, height, and color, KNN can classify a new animal without assuming a specific distribution (like normal distribution or frequency distribution) of the data. 
<br>
Instead, it directly uses the data points to make decisions based on their relationships.

## Classification Problems
KNN is **mainly** used for **classification** problems. For example, in a handwritten digit recognition task, each digit (0-9) can be represented as an image. When a new digit image is inputted, KNN looks at the k nearest digit images in the training set. If, for instance, (k = 5) the closest **five images** consist of three '7's and two '9's, the algorithm assigns the label '7' to the new image based on majority voting.

## Usage in Regression Problems
KNN can also be used for regression tasks. For instance, imagine predicting the price of houses based on features like size and location. When a new house data point is introduced, KNN will find the k nearest houses and calculate the average price of those houses. If the nearest three houses sold for $300,000, $350,000, and $400,000, the algorithm would predict the new house's price as approximately $350,000, which is the average of the neighbors' prices.


## Lazy Learning : (Very Important)  
KNN is classified as a lazy learning algorithm because it doesn't have a **training phase**; it simply stores the **entire training dataset**. For example, imagine KNN as a data analyst who keeps all customer information in a database instead of summarizing it into a model. When a new customer inquiry comes in, the analyst queries the database to find similar customers based on their characteristics. 
<br>
This method allows for flexible responses, but it can be **slow** because the analyst has to search through **all the records** **each time a new inquiry** is made.



