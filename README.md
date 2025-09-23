# Unsupervised-Learning-Recommenders-Reinforcement-Learning

The concept of clustering in unsupervised learning.

Clustering Overview

A clustering algorithm identifies and groups similar data points without prior labels.
Unlike supervised learning, which uses labeled data to predict outcomes, clustering seeks to find structure in unlabeled data.
Applications of Clustering

Clustering can be used to group similar news articles or segment markets based on learner interests.
It is also applied in analyzing DNA data to group individuals with similar genetic traits.
Examples in Astronomy

Astronomers utilize clustering to analyze astronomical data, grouping celestial bodies to understand their relationships and structures in space.

The K-means clustering algorithm is a method used to group data points into clusters based on their similarities.

Initial Cluster Centroids

The algorithm starts by randomly selecting initial points as the centers of the clusters, known as cluster centroids.
In this example, two centroids are chosen, represented by a red cross and a blue cross.
Assigning Points to Clusters

Each data point is evaluated to determine which centroid it is closest to, and is then assigned to that cluster.
Points are visually represented by colors (red or blue) based on their assigned cluster.
Updating Cluster Centroids

After assigning points, the algorithm calculates the average position of all points in each cluster to update the centroids' locations.
This process is repeated: points are reassigned to the nearest centroid, and centroids are updated based on the new assignments.
Convergence of the Algorithm.

The K-means algorithm is a popular clustering method used in unsupervised learning. 

Initialization of Cluster Centroids

The algorithm begins by randomly initializing K cluster centroids (Mu 1, Mu 2, ..., Mu k).
Each centroid corresponds to a cluster, and they are represented as vectors with the same dimensions as the training examples.
Assignment of Points to Clusters

The first step involves assigning each training example to the nearest cluster centroid based on distance.
This is done by calculating the distance between each training example and the centroids, typically using the L2 norm.
Updating Cluster Centroids

The second step updates the position of each centroid to the mean of the points assigned to that cluster.
If a cluster ends up with no points assigned, it is common to either eliminate that cluster or reinitialize its centroid.
Application of K-means

K-means can be applied even when clusters are not well-separated, such as in sizing t-shirts based on customer height and weight data.
The algorithm aims to optimize a specific cost function, which will be explored in the next video.

The K-means algorithm and its optimization process through a specific cost function.

Cost Function of K-means

The cost function J measures the average squared distance between training examples and their assigned cluster centroids.
The notation used includes cluster indices and centroid locations, which help in understanding the assignments of training examples to clusters.
Steps of the K-means Algorithm

The first step involves assigning points to the nearest cluster centroid to minimize the cost function while keeping centroids fixed.
The second step updates the cluster centroids by calculating the mean of the points assigned to each cluster, aiming to minimize the cost function.
Convergence of the Algorithm

The K-means algorithm is guaranteed to converge, as each iteration either reduces or maintains the value of the cost function.
If the cost function stops decreasing, it indicates convergence, and the algorithm can be terminated.
Utilizing Random Initialization

Using multiple random initializations of cluster centroids can lead to better clustering results, enhancing the effectiveness of the K-means algorithm.

The initial steps of the K-means clustering algorithm, particularly how to choose random initial guesses for cluster centroids.

Choosing Initial Centroids

The first step in K-means is to select random locations for the cluster centroids (mu1 through muK).
A common method is to randomly pick K training examples from the dataset to serve as the initial centroids.
Impact of Random Initialization

Different random initializations can lead to different clustering results, sometimes resulting in local optima.
Running K-means multiple times with different initializations can help find a better clustering solution.
Evaluating Clustering Quality

After running K-means multiple times, the cost function J can be computed for each clustering result.
The clustering with the lowest cost function value is typically chosen as the best solution, indicating a better fit to the data.
Algorithm for Multiple Initializations

The recommended approach is to run K-means multiple times (e.g., 50 to 1000) with different random initializations.
This method increases the likelihood of finding a more optimal clustering solution by minimizing the distortion cost function.

The challenges of determining the optimal number of clusters (k) in the k-means clustering algorithm.

Understanding the Ambiguity of Clusters

The right value of k is often ambiguous; different observers may identify different numbers of clusters in the same dataset.
Clustering is an unsupervised learning method, meaning there are no predefined labels to guide the clustering process.
Techniques for Choosing k

The elbow method is one technique where k-means is run with various k values, and the cost function is plotted to identify a point where the decrease in cost slows down, resembling an "elbow."
However, the elbow method may not always yield a clear choice, as many datasets do not exhibit a distinct elbow.
Practical Considerations for k

The choice of k should be based on the specific application and the trade-offs involved, such as the balance between fit and cost.
An example is provided with t-shirt sizing, where different k values can lead to different groupings, and the decision should consider business implications.
The content concludes by introducing the next topic, anomaly detection, as another important application of unsupervised learning.

Anomaly detection is a key unsupervised learning algorithm that identifies unusual events in a dataset of normal occurrences. 

Understanding Anomaly Detection

Anomaly detection algorithms learn from unlabeled datasets to flag unusual events, such as defects in manufactured products like aircraft engines.
The algorithm analyzes features (e.g., heat and vibration) of normal engines to determine if a new engine is functioning properly.
How Anomaly Detection Works

The algorithm uses density estimation to model the probability of feature values, identifying regions of high and low probability.
If a new engine's feature vector has a low probability (below a threshold), it is flagged as an anomaly for further inspection.
Applications of Anomaly Detection

Commonly used in fraud detection to monitor user behavior and identify suspicious activities.
Employed in manufacturing to ensure products meet quality standards before shipping.
Utilized in monitoring computer systems to detect failures or security breaches.
Anomaly detection is a widely applicable tool in various industries, helping to maintain quality and security. The next steps involve learning how to implement these algorithms using Gaussian distributions.

The Gaussian distribution, also known as the normal distribution, which is essential for anomaly detection.

Understanding Gaussian Distribution

The Gaussian distribution is characterized by its bell-shaped curve, defined by two parameters: mean (Mu) and variance (Sigma squared).
The mean determines the center of the curve, while the standard deviation (Sigma) affects its width.
Effects of Changing Parameters

Adjusting Mu shifts the curve left or right, while changing Sigma alters the curve's width and height.
A smaller Sigma results in a narrower and taller curve, while a larger Sigma creates a wider and shorter curve.
Application in Anomaly Detection

To apply this distribution in anomaly detection, one estimates Mu and Sigma squared from a dataset.
The formulas for estimating these parameters are based on the average of training examples and the average of squared differences from the mean.
In summary, understanding the Gaussian distribution and its parameters is crucial for effectively applying anomaly detection techniques in datasets.

Building an anomaly detection algorithm using Gaussian (normal) distribution.

Understanding the Training Set

The training set consists of examples with multiple features, represented as vectors.
Each feature vector is modeled to estimate the probability of its occurrence, denoted as p(x).
Modeling Probability

The probability p(x) is calculated as the product of individual feature probabilities, assuming statistical independence.
Each feature's probability is modeled using two parameters: mean (mu) and variance (sigma squared).
Anomaly Detection Process

Features indicative of anomalies are selected, and parameters are estimated from the training set.
A new example is evaluated by computing p(x) and comparing it to a threshold (epsilon) to determine if it is an anomaly.
Example Application

The algorithm flags an example as anomalous if one or more features are significantly different from the training set.
A practical example illustrates how to compute p(x) for test cases and identify anomalies based on the computed probabilities.

Practical tips for developing an anomaly detection system, emphasizing the importance of evaluation during the development process.

Evaluation in Anomaly Detection

Real number evaluation allows for quick adjustments to the learning algorithm, making it easier to decide on changes to features or parameters.
Using a small number of labeled anomalies (y=1) alongside a larger set of normal examples (y=0) helps in evaluating the algorithm's performance.
Training and Validation Sets

A typical setup includes a training set with normal examples, a cross-validation set with both normal and a few anomalous examples, and a test set for final evaluation.
The training set can have some mislabeled anomalies, but the algorithm can still perform adequately.
Algorithm Evaluation

After fitting the model, predictions are made based on the computed probabilities, determining if an example is anomalous or normal.
Metrics like true positive, false positive, and precision-recall are useful for evaluating performance, especially in skewed data distributions.
Challenges with Limited Data

In cases with very few anomalies, it may be necessary to combine the cross-validation and test sets, which increases the risk of overfitting.
The content concludes by suggesting that having a few labeled anomalies simplifies the evaluation process, leading to better tuning of the algorithm.

The decision-making process between using anomaly detection and supervised learning algorithms based on the availability of positive and negative examples.

Anomaly Detection vs. Supervised Learning

Anomaly detection is preferred when there are very few positive examples (0-20) and many negative examples, focusing on modeling normal behavior to identify deviations.
Supervised learning is more suitable when there are sufficient positive and negative examples, assuming future examples will resemble those in the training set.
Examples of Applications

Anomaly detection is effective in scenarios like financial fraud detection, where new fraud types frequently emerge, making it hard to learn from past examples.
Supervised learning works well for tasks like email spam detection, where spam types tend to be similar over time, allowing the model to learn from previous examples.
Feature Importance in Anomaly Detection

The choice of features is crucial when building anomaly detection systems, as it influences the model's ability to identify new anomalies effectively.

The importance of feature selection in building effective anomaly detection algorithms.

Feature Selection in Anomaly Detection

Choosing the right features is crucial for anomaly detection, more so than in supervised learning, as the algorithm learns from unlabeled data.
Features should ideally follow a Gaussian distribution; transformations like logarithmic or square root can help achieve this.
Transforming Features

Plotting histograms of features can reveal their distribution; non-Gaussian features may need transformation.
Common transformations include log(X), log(X + C), and X^0.5, which can help make the data more Gaussian.
Error Analysis and Feature Creation

If the model struggles with certain anomalies, analyzing these errors can inspire new feature creation.
New features can help distinguish anomalies by capturing unusual patterns, improving the algorithm's performance.
Overall, the process involves training the model, analyzing errors, and iteratively refining features to enhance anomaly detection capabilities.

The algorithm continues to iterate through the assignment and updating steps until there are no changes in point assignments or centroid locations.
At this point, the algorithm has converged, effectively identifying the clusters within the data set.

The concept of recommender systems, which are widely used in various online platforms to suggest products, movies, or services to users based on their preferences.

Understanding Recommender Systems

Recommender systems analyze user ratings and preferences to suggest items they may like.
They are crucial for businesses, as a significant portion of sales can be attributed to these systems.
Example of Movie Ratings

The example illustrates a scenario with users rating movies on a scale of one to five stars.
The system tracks which users have rated which movies and uses this data to predict ratings for unrated movies.
Algorithm Development

The course will explore how to develop algorithms to predict user ratings for items they haven't rated yet.
Initially, the focus will be on using additional features about the items (like movie genres) to enhance recommendations.
This foundational understanding will set the stage for further exploration of recommender systems in the upcoming lessons.

Developing a recommender system using features of items, specifically movies, to predict user ratings.

Understanding Movie Features

Two features, X1 and X2, are introduced to describe movies: romance and action.
Each movie is assigned values for these features, indicating their characteristics (e.g., "Love at Last" is highly romantic).
Predicting User Ratings

The prediction for a user's rating is modeled similarly to linear regression, using parameters w and b.
Different parameters are used for each user, allowing personalized predictions based on their previous ratings.
Cost Function and Optimization

A cost function is formulated to minimize the error between predicted and actual ratings, focusing only on movies rated by the user.
Regularization is added to prevent overfitting, and the overall cost function is summed across all users to learn parameters effectively.
This approach allows for personalized movie recommendations based on user preferences and movie features.

How to derive features for movies in a collaborative filtering context when those features are not initially known.

Understanding Features in Collaborative Filtering

The discussion begins with the concept of predicting movie ratings using linear regression, where features like romance and action levels (x_1 and x_2) are used.
It highlights the challenge of not having predefined features and introduces the idea of estimating these features from user ratings.
Learning Features from User Ratings

The example illustrates how to guess reasonable features for a movie based on user ratings and learned parameters (w and b).
A cost function is proposed to minimize the squared error between predicted and actual ratings, allowing for the learning of features for each movie.
Combining Learning of Parameters and Features

The content explains how to combine the cost functions for learning user parameters (w and b) and movie features (x) into a single collaborative filtering algorithm.
It emphasizes the importance of multiple user ratings in deriving features, which is a key advantage of collaborative filtering over traditional linear regression methods.

Generalizing collaborative filtering algorithms to work with binary labels in recommender systems.

Understanding Binary Labels

Binary labels indicate whether a user liked (1) or did not like (0) an item, with a question mark (?) representing items not yet seen.
Examples include online shopping (purchase or not) and social media (like or not).
Generalizing the Algorithm

The algorithm is adapted from linear regression to logistic regression to predict the probability of a user liking an item.
The logistic function is used to model the probability, transforming the linear prediction into a logistic regression model.
Cost Function Modification

The cost function is changed from squared error to binary cross-entropy, suitable for binary labels.
The new cost function sums over all user-item pairs, focusing on the predicted probabilities and actual binary labels.
This adaptation allows for a broader range of applications in recommender systems, enhancing their effectiveness.

The concept of mean normalization in the context of building a recommender system, particularly for movie ratings.

mean normalization in recommender systems

Mean normalization helps algorithms run more efficiently and improves predictions for users who have not rated any movies.
By normalizing ratings to have a consistent average, the algorithm can make better predictions for new users, like Eve, who has no prior ratings.
calculating mean ratings

The average rating for each movie is calculated based on existing ratings, creating a vector of average ratings (μ).
Ratings are adjusted by subtracting the mean rating for each movie, resulting in new values that allow for better predictions.
impact on predictions

The adjusted ratings help predict more reasonable ratings for new users, avoiding the assumption that they will rate all movies with zero stars.
Normalizing the rows of the rating matrix is emphasized as more beneficial for new users compared to normalizing columns, which is less critical for movies with no ratings.

Implementing the collaborative filtering algorithm using TensorFlow, highlighting its capabilities beyond building neural networks.

Understanding TensorFlow's Role

TensorFlow can automatically compute derivatives of cost functions, simplifying the implementation of gradient descent.
Users only need to define the cost function, allowing TensorFlow to handle the calculus involved.
Gradient Descent with TensorFlow

The lecture revisits the gradient descent update process, emphasizing the importance of the derivative term.
TensorFlow's gradient tape feature records operations to enable automatic differentiation, making it easier to optimize parameters.
Implementing Collaborative Filtering

The collaborative filtering algorithm can be implemented using TensorFlow's tools, including the Adam optimizer for more efficient optimization.
The cost function for collaborative filtering requires specific inputs, and TensorFlow can compute the necessary derivatives automatically, streamlining the process.

The collaborative filtering algorithm used in online shopping to recommend similar items to users.

Collaborative Filtering Overview

The algorithm identifies features of items (e.g., movies or books) to recommend similar products based on user preferences.
It calculates the squared distance between feature vectors of items to find those that are most similar.
Limitations of Collaborative Filtering

Cold Start Problem: New items or users with few ratings can lead to inaccurate recommendations.
Lack of Side Information: The algorithm does not effectively utilize additional data about items or users, such as demographics or preferences.
Next Steps

The content hints at exploring content-based filtering algorithms in the following video, which can address some limitations of collaborative filtering.

The development of a content-based filtering algorithm for recommender systems, contrasting it with collaborative filtering.

Collaborative Filtering vs. Content-Based Filtering

Collaborative filtering recommends items based on ratings from similar users.
Content-based filtering recommends items based on user and item features, aiming for better matches.
User and Item Features

User features may include age, gender, country, and past behaviors (e.g., movies watched).
Item features can include movie genre, year, critic reviews, and average ratings.
Vector Representation

Each user and item is represented by a feature vector (vj_u for users and vi_m for items).
The dot product of these vectors predicts the rating a user would give to an item.
In summary, content-based filtering utilizes user and item features to create vectors that help in making personalized recommendations, differing from the collaborative approach that relies solely on user ratings.

The development of a content-based filtering algorithm using deep learning techniques.

User Network and Movie Network

A user network processes user features (e.g., age, gender) to produce a user vector ( v_u ) with 32 units.
A movie network processes movie features (e.g., year of release, stars) to produce a movie vector ( v_m ).
Prediction and Cost Function

The predicted rating is calculated as the dot product of ( v_u ) and ( v_m ).
A cost function ( J ) is constructed to minimize the squared error between predicted ratings and actual ratings, using gradient descent for training.
Finding Similar Items

The trained model can also identify similar items by measuring the distance between movie vectors.
Pre-computation of similar items can enhance user experience by quickly providing recommendations.
Complexity and Feature Engineering

Combining user and movie networks allows for a more complex architecture.
Careful feature engineering is essential for effective implementation, especially in large catalogs.

The efficient functioning of recommender systems, particularly in handling large catalogs of items.

Retrieval and Ranking Steps

Recommender systems typically operate in two main steps: retrieval and ranking.
The retrieval step generates a broad list of potential item candidates, which may include items the user may not prefer.
Retrieval Process

For each of the last 10 items a user interacted with, the system finds similar items, creating an initial list of recommendations.
Additional items can be added based on the user's preferred genres or popular items in their region.
Ranking Process

The ranking step refines the list by predicting user ratings for the retrieved items using a neural network.
The system computes predicted ratings for each user-item pair and displays the highest-rated items to the user.
Optimization Considerations

The number of items retrieved impacts performance; more items can lead to better recommendations but may slow down the process.
Offline experiments can help determine the optimal number of items to retrieve for improved relevance in recommendations.
Ethical Considerations

The content concludes by highlighting the importance of ethical considerations in building recommender systems, emphasizing the need to serve users and society responsibly.

The implications and challenges of recommender systems, particularly focusing on their potential negative impacts on society.

Problematic Use Cases

Recommender systems can prioritize profit over user benefit, leading to recommendations that may not align with users' best interests.
Examples include recommending ads based on profitability rather than relevance, which can exploit users.
Advertising Dynamics

The advertising industry can amplify both beneficial and harmful businesses, creating positive or negative feedback loops.
A good example is the travel industry, where quality service leads to profitability, while the payday loan industry often exploits vulnerable customers.
User Engagement Concerns

Maximizing user engagement can lead to the amplification of harmful content, such as conspiracy theories and hate speech.
Companies face challenges in defining and filtering out problematic content while maintaining user trust.
Transparency and Ethical Considerations

Users often assume recommendations are based on their preferences, not realizing profit motives may drive them.
Encouraging transparency in recommendation criteria can help build trust and promote ethical practices in AI development.

Implementing content-based filtering using TensorFlow, particularly in the context of recommender systems.

User and Item Networks

The implementation begins with defining a user network and an item network (movies as items) using a sequential model with dense layers.
Each network consists of two dense hidden layers, with the final layer outputting 32 units.
Feature Feeding and Normalization

TensorFlow Keras is instructed on how to feed user and item features into the respective networks.
An important step is normalizing the user and item vectors to have a length of one, enhancing the algorithm's performance.
Dot Product and Model Definition

The dot product between the user and item vectors is computed using a special Keras layer designed for this purpose.
The model's inputs and outputs are defined, with the mean squared error cost function selected for training.
Overall, the content provides key code snippets for implementing a content-based filtering algorithm in TensorFlow, emphasizing the importance of normalization for better performance.

Principal Components Analysis (PCA), an unsupervised learning algorithm used for data visualization.

Understanding PCA

PCA helps reduce high-dimensional data (e.g., 50 or 1,000 features) to two or three dimensions for easier visualization.
It identifies the most significant features that capture the variance in the data.
Examples of PCA Application

In a dataset of passenger cars, PCA can determine which features (like length or width) are most informative for visualization.
For complex datasets, PCA can create new axes (e.g., combining length and height) to summarize information effectively.
Practical Use of PCA

PCA is commonly used to visualize data from various fields, such as economics, where multiple features (like GDP and Human Development Index) can be reduced to two dimensions for analysis.
This technique aids in understanding data patterns and identifying anomalies or unexpected trends.
