> **Machine Learning (ML)** is a branch of Artificial Intelligence where computers **learn patterns from data** and use those patterns to **make predictions or decisions**, without being explicitly programmed for every situation.

#### Traditional Programming
```
        Rules
          ↓
Input → PROGRAM → Output
```
#### Machine Learning
```
        Data + Answers
              ↓
        ML ALGORITHM
              ↓
        Trained MODEL
              ↓
       New / Unseen Data
              ↓
          Prediction
```

==Instead of manually writing all the rules, we give the algorithm **examples**, and it learns the underlying pattern.==

#### What exactly does a model "learn"?
A model does **not** learn like a human.
It learns **mathematical patterns/parameters** from data.

For example, a simple model might learn:
y = wx + b

Where:
- `x` → input
- `y` → prediction
- `w` → learned weight
- `b` → learned bias

During training, ML tries to find values of `w` and `b` that make predictions as accurate as possible.

# TYPES OF MACHINE LEARNING

Machine Learning is mainly classified based on ==**how the model learns from data**.==
```
                    MACHINE LEARNING
                          │
          ┌───────────────┼────────────────┐
          ↓               ↓                ↓
    Supervised       Unsupervised     Reinforcement
      Learning          Learning          Learning
          │               │                │
     Has Labels       No Labels       Reward / Penalty
          │               │                │
      ┌───┴───┐       ┌───┴────┐           │
      ↓       ↓       ↓        ↓           ↓
 Regression Classification Clustering  Dimensionality   Agent
                                       Reduction
```

The **three major categories** you should know first are:
1. **Supervised Learning**
2. **Unsupervised Learning**
3. **Reinforcement Learning**

There are also important extensions such as **Semi-Supervised Learning** and **Self-Supervised Learning**, which become particularly relevant later in modern AI.

### 1. Supervised Learning 

> **Supervised Learning** is ML where a model learns from **labeled data**, meaning each training example contains both the input and the desired output.

```
Input (X) ──────→ Model ──────→ Prediction (ŷ)
                     ↑
                  Training
                     ↑
              Actual Output (y)
```

#### Example
Predict house price:

|Area|Bedrooms|Price|
|---|---|---|
|1000|2|₹50L|
|1500|3|₹70L|
|2000|3|₹90L|

Here:
```
Features → Area, Bedrooms
Label    → Price
```

Because we **know the correct answer during training**, this is supervised learning.

#### Two major types

##### A. Regression

> Predicts a **continuous numerical value**.

Examples:
- House price → ₹75L
- Temperature → 32.5°C
- Salary → ₹8.2L
- Stock demand → 15,430 units

```
Input → Model → Number
```

##### B. Classification

> Predicts a **category/class**.

Examples:
```
Email → Spam / Not Spam
Image → Cat / Dog
Transaction → Fraud / Genuine
```

```
Input → Model → Class
```


### 2. Unsupervised Learning 

> **Unsupervised Learning** is ML where the model receives **unlabeled data** and attempts to discover useful patterns, structures, or relationships within it.

There is **no predefined correct answer**.

```
             Unlabeled Data
                   ↓
                 Model
                   ↓
          Patterns / Structure
```

#### Example: Customer Segmentation
Suppose we have:
```
Customer
├── Age
├── Income
├── Spending
└── Purchase frequency
```

We don't tell the model:
> "These are premium customers."

Instead, the model may discover:
```
             Customers
                 │
        ┌────────┼────────┐
        ↓        ↓        ↓
     Group A   Group B   Group C
     Low       Medium    High
     Spending Spending  Spending
```

This is **clustering**.

#### Important Unsupervised Tasks
##### A. Clustering
Groups similar data points together.

Examples:
- Customer segmentation
- Document grouping
- Image grouping

Common algorithm:
**K-Means**

##### B. Dimensionality Reduction
Reduces the number of features while attempting to preserve important information.

Example:
```
100 Features
     ↓
Dimensionality Reduction
     ↓
2 / 3 Features
```

Useful for:
- Visualization
- Noise reduction
- Faster processing
- Feature representation

Common techniques:
- PCA
- t-SNE
- UMAP

### 3. Reinforcement Learning 

> **Reinforcement Learning (RL)** is a learning paradigm where an **agent interacts with an environment**, takes actions, and learns from **rewards or penalties**.

The agent's objective is to learn a **policy** that maximizes cumulative reward over time.

###mple: Game AI

Imagine an AI playing chess.
```
AI
 ↓
Makes a move
 ↓
Game state changes
 ↓
Gets reward / penalty
 ↓
Learns which actions are better
```

The model isn't simply given the correct move for every situation.
It **learns through interaction and feedback**.

### 4. Semi-Supervised Learning
A combination of supervised and unsupervised learning.

> Uses **a small amount of labeled data + a large amount of unlabeled data**.

```
Small labeled dataset
        +
Large unlabeled dataset
        ↓
   ML Algorithm
        ↓
     Model
```

#### Why?
Labeling data can be expensive.
For example, suppose you have:
```
10,000,000 images
```

but only:
```
10,000 labeled images
```

Semi-supervised methods can potentially use the remaining unlabeled images to improve learning.

### 5. Self-Supervised Learning 
This becomes **very important for modern AI**.

> **Self-Supervised Learning** creates training signals from the data itself rather than requiring humans to manually provide labels.

Example: predicting the next word:
```
"The cat is sitting on the ___"

                    ↓
                 "mat"
```

The original text itself provides the training signal.

This idea is fundamental to how **large language models** are pretrained.
```
Huge Dataset
     ↓
Create learning task from data
     ↓
Neural Network
     ↓
Learn representations
     ↓
Pretrained Model
```

It Consists in **Deep Learning → Transformers → LLMs**.

# Steps involved in making a ML Model

![[Pasted image 20260905121700.png]]

### Exploratory Data Analysis (EDA)

> **Exploratory Data Analysis (EDA)** is the process of **examining, summarizing, and visualizing a dataset** to understand its structure, distributions, relationships, patterns, and potential problems before applying a machine learning model.

> **EDA = Understand your data before making the model learn from it.**

```
Raw Dataset
     ↓
    EDA
     ↓
Understand:
• What data do I have?
• Is it clean?
• What patterns exist?
• Are there missing values?
• Are there outliers?
• Which features matter?
     ↓
Better preprocessing + modeling decisions
```

> **EDA doesn't primarily change the data — it helps you understand what needs to be changed.**

#### Usage of EDA

##### Duplicate Data
Check whether the same observation appears multiple times.

```
Row 101 = Row 450
        ↓
Possible duplicate
```

##### Invalid Values

Example:
```
Age:

21
25
34
-10  ← Invalid
500  ← Suspicious
```
EDA helps identify such problems.

##### Outliers
An **outlier** is an observation that is unusually far from the rest of the data.

Example:
```
Salary:

₹40k
₹45k
₹50k
₹52k
₹48k
₹5 crore  ← Outlier
```

Common ways to detect them:
- Box plots
- Histograms
- IQR
- Z-score

# Data Cleaning & Data Preprocessing
## Data Cleaning

> **Data Cleaning** is the process of identifying and fixing **errors, missing values, duplicates, and inconsistencies** in a dataset.

#### Common tasks
```
Raw Data
   ↓
Missing Values ──→ Handle
Duplicates ──────→ Remove
Invalid Values ──→ Correct
Outliers ────────→ Analyze / Handle
Inconsistencies → Fix
   ↓
Clean Data
```

| Age | Salary |
| --- | ------ |
| 21  | 50000  |
| —   | 60000  |
| 25  | 50000  |
| -10 | 70000  |

Possible cleaning:
- Missing age → impute/remove
- Duplicate row → remove if it is truly a duplicate
- `-10` age → investigate and correct/remove
## Data Preprocessing 

> **Data Preprocessing** is the broader process of **transforming raw data into a format suitable for machine learning algorithms**.

It includes cleaning **plus transformations**.
#### Common steps
```
Raw Data
   ↓
Cleaning
   ↓
Encoding
   ↓
Scaling / Normalization
   ↓
Feature Transformation
   ↓
ML-Ready Data
```

### Important techniques
#### 1. Missing Value Handling
- Mean / Median / Mode
- Imputation
- Removing records

#### 2. Encoding Categorical Data
##### A. Label Encoding
> Converts **categories into numerical labels**.

Example:
```
Small  → 0
Medium → 1
Large  → 2
```
#### ⚠️ Problem
The numbers may imply an order.
```
Red   → 0
Blue  → 1
Green → 2
```
There is **no actual numerical relationship** between colors.

**Best for:** Ordinal categories where order matters.
```
Low < Medium < High
  0      1       2
```

##### B. One-Hot Encoding 
> Converts each category into a **separate binary feature (0/1)**.

Example:
```
Color
Red
Blue
Green
```

becomes:

|Red|Blue|Green|
|---|---|---|
|1|0|0|
|0|1|0|
|0|0|1|
Used Categories have **no natural order**.
```
Color, City, Country, Browser
```

#### 3. Feature Transformation
> Changing the **representation or distribution of a feature** to make it more suitable for ML.

Examples:
```
Date → Year, Month, Day
Salary → log(Salary)
Height in cm → Height in m
```

#### 4. Feature Scaling 
> Bringing numerical features to a **comparable scale**.

Example:
```
Age     → 18–80
Salary  → 20,000–2,00,000
```
Salary has much larger numerical values and can dominate some algorithms.

Scaling solves this:
```
Age     → comparable scale
Salary  → comparable scale
```

Two important methods:
```
          Feature Scaling
                │
         ┌──────┴──────┐
         ↓             ↓
 Normalization   Standardization
```

##### A. Normalization
> Usually scales values to a fixed range, commonly **0 to 1**.

##### Min-Max Normalization
![[Pasted image 20260905124841.png]]

Example:
```
Original:  20, 30, 40
                ↓
Normalized: 0, 0.5, 1
```
##### Useful when:
- Features have different ranges
- Algorithms are sensitive to scale
- Neural networks / distance-based methods

##### B. Standardization 
> Transforms data so that it has approximately **mean = 0 and standard deviation = 1**.

![[Pasted image 20260905124857.png]]

Where:
- x = original value
- μ = mean
- σ = standard deviation

Example:
```
Original values
      ↓
Standardization
      ↓
Mean ≈ 0
Std  ≈ 1
```
Unlike normalization, values **do not have to lie between 0 and 1**.

# Feature Engineering & Feature Selection
## Feature Engineering 

> **Feature Engineering** is the process of **creating, transforming, or combining features** to make them more useful for a machine learning model.

#### Example
Suppose we have:
```
Date of Birth → 15/05/2000
Current Date → 05/09/2026
```

Instead of giving the raw dates to the model:
```
Date of Birth
      ↓
Feature Engineering
      ↓
Age = 26
```

Another example:
```
Total Amount = ₹10,000
Number of Orders = 5

      ↓

Average Order Value = ₹2,000
```

#### Common techniques
- Creating new features
- Combining existing features
- Extracting information from dates/text
- Log/polynomial transformations
- Binning continuous values
### Goal
> **Create better input features → help the model learn better patterns.**

## Feature Selection

> **Feature Selection** is the process of choosing the **most relevant features** from the existing dataset and removing unnecessary ones.

Example:
```
20 Features
     ↓
Feature Selection
     ↓
8 Important Features
```

Suppose predicting house price:
```
Area          ✓
Bedrooms      ✓
Location      ✓
Age           ✓
Shoe Size     ✗
Favorite Color ✗
```

The irrelevant features can add **noise**, increase complexity, and sometimes hurt performance.

### Common Feature Selection Methods

##### ① Filter Methods
Select features using statistical measures.

Examples:
- Correlation
- Chi-square
- Mutual information
##### ② Wrapper Methods
Try different feature subsets and evaluate model performance.

Example:
```
Features A+B+C → Model → Score
Features A+B   → Model → Score
Features A+C   → Model → Score
```

##### ③ Embedded Methods
Feature selection happens **during model training**.

Examples:
- Lasso (L1 regularization)
- Decision Tree feature importance

# REGRESSION

> **Regression is a supervised learning technique used to predict a continuous numerical value from one or more input features.**
#### Example
```
Area + Bedrooms + Location
            ↓
      Regression Model
            ↓
       House Price
        ₹85,00,000
```

Other examples:
- Predict **salary**
- Predict **temperature**
#### How Regression Works
A regression model tries to learn the relationship between:
```
Input Features (X)
       ↓
    MODEL
       ↓
Predicted Value (ŷ)
```

For simple linear regression:

y = mx + c

Where:
- x → input feature
- y → predicted value
- m → slope/weight
- c → intercept/bias
The model learns the parameters so that its predictions are as close as possible to the actual values.

## 1. Linear Regression

> **Linear Regression** finds a straight-line relationship between input feature(s) and a continuous target.

![[Pasted image 20260905182346.png]]
#### Example
```
Area (X) ─────→ Linear Regression ─────→ Price (Y)
```

The model tries to find the line (Best Fit Line) that represents the data **as closely as possible**.

### Scatter Plot
A **scatter plot** represents individual observations as points.

Example:
```
Price ↑
      |                 •
      |            •
      |         •
      |      •
      |   •
      | •
      └────────────────────→ Area
```

Each point represents:
(xi,yi)

==A scatter plot helps us see whether a **linear relationship** may exist between two numerical variables.==

### Best-Fit Line 
The **best-fit line** is the line that minimizes the overall squared difference between
actual and predicted values.

![[Pasted image 20260905182901.png|611]]

The Blue one is the Data :
![[Pasted image 20260905182735.png|268]]

The Green one is the Best Fit Line over The prediction Points:
![[Pasted image 20260905182838.png|271]]

##### **NOTE:**
![[Pasted image 20260905183150.png|477]]


### Residual Error 

> **Residual = Actual value (Original Data Point) − Predicted value (Predicted Data Point)**

![[Pasted image 20260905183227.png]]

![[Pasted image 20260905184212.png|413]]

Example:
```
Actual price     = ₹80L
Predicted price  = ₹75L

Residual = 80 - 75
         = ₹5L
```

#### Important
- Residual = **0** → prediction is exact
- Positive residual → model **underpredicted**
- Negative residual → model **overpredicted**

### Cost Function 

> **Cost Function** measures how far the model's predictions are from the actual values.

![[Pasted image 20260905183336.png|572]]

![[Pasted image 20260905184135.png|554]]
### Gradient Descent

> ==**Gradient Descent** is an optimization algorithm used to **minimize the cost function** by repeatedly updating the model's parameters in the direction of decreasing cost.==

![[Pasted image 20260905183627.png|509]]
#### Intuition
```
High Cost
   ●
    \
     \
      ●
       \
        ●  ← Minimum Cost
```

The gradient tells us **which direction the cost increases**.
So we move in the **opposite direction**.

### Global Minimum Point 
> The **global minimum** is the point where the cost function has its **lowest possible value**.

![[Pasted image 20260905184035.png|380]]

At the global minimum:
![[Pasted image 20260905183711.png]]

For a **convex cost function**, such as the standard Linear Regression MSE cost function, the local minimum is also the global minimum.

### Repeat Until Convergence
Gradient Descent doesn't usually reach the minimum in one step.

It repeatedly performs:
```
Initialize parameters
        ↓
  Calculate Cost
        ↓
 Calculate Gradient
        ↓
 Update Parameters
        ↓
      Repeat
        ↓
   Convergence
```

#### Convergence
> **Convergence occurs when further iterations produce little or no meaningful improvement in the cost function.**

For example:
```
Cost:
100 → 60 → 35 → 20 → 15 → 14.9 → 14.9
                                      ↑
                                  Converged
```
#### Learning Rate matters
- Too small → very slow convergence
- Too large → may overshoot or diverge
- Appropriate → reaches minimum efficiently

### Hyperplane

> A **hyperplane** is a flat decision/prediction surface in a higher-dimensional space.

#### For Linear Regression:
##### 1 feature
A **line**:
![[Pasted image 20260905184522.png]]
##### 2 features
A **plane**:
![[Pasted image 20260905184539.png]]
##### Many features
A **hyperplane**:
![[Pasted image 20260905184550.png]]

> A hyperplane is simply the **higher-dimensional generalization of a line or plane**.

