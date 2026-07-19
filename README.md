A machine learning project that implements Linear Regression from scratch using Gradient Descent to predict student academic performance based on previous exam scores.

Built with Python, NumPy, and Matplotlib, this project demonstrates the complete workflow of a supervised machine learning model, including data preprocessing, feature normalization, gradient descent optimization, model evaluation, and data visualization.

Project Overview

Linear Regression is one of the most fundamental algorithms in machine learning and serves as the foundation for understanding predictive modeling. Rather than relying on machine learning libraries such as Scikit-Learn to train the model, this project manually implements the mathematics behind Linear Regression using Gradient Descent.

The application loads a real-world student performance dataset, selects Previous Scores as the predictor variable, normalizes the feature for faster convergence, and iteratively learns the optimal regression line by minimizing the Mean Squared Error (MSE).

Throughout the training process, the model tracks the loss at each iteration and visualizes both the fitted regression line and the error convergence, providing a clear understanding of how Gradient Descent improves the model over time.

Features
Linear Regression implemented entirely from scratch
Gradient Descent optimization
Mean Squared Error (MSE) calculation
Automatic feature normalization
Vectorized NumPy implementation for improved performance
Regression line visualization
Error convergence graph
Automatic visualization export
Technologies Used
Python
NumPy
Matplotlib
Machine Learning Pipeline
1. Data Loading

The application loads the Student Performance Dataset from a CSV file using NumPy. Only the relevant columns are extracted:

Previous Scores (Independent Variable)
Performance Index (Target Variable)

Rows containing missing values are automatically removed before training begins.

2. Data Preprocessing

To improve Gradient Descent convergence, the input feature is normalized using standardization.

The feature's mean and standard deviation are calculated, allowing the data to be scaled without affecting the relationship between the variables.

3. Model Training

The model begins with randomly initialized parameters for the slope and intercept.

During each iteration of Gradient Descent, the algorithm:

Predicts student performance
Computes prediction errors
Calculates the gradients
Updates the regression parameters
Stores the Mean Squared Error

This process repeats until the model converges toward the optimal regression line.

4. Error Calculation

The performance of the model is measured using Mean Squared Error (MSE).

The loss value is recorded after every training iteration, allowing the convergence behavior of Gradient Descent to be visualized.

5. Data Visualization

After training, the application generates two visualizations:

Scatter plot of the original student data with the fitted regression line
Error convergence graph showing how the Mean Squared Error decreases throughout training

The visualization is automatically saved as:

regression_visualization.png
6. Model Output

Once training is complete, the application prints:

Initial Mean Squared Error
Final Mean Squared Error
Learned slope
Learned intercept
Final regression equation
Repository Structure
Student-Performance-Linear-Regression/
│
├── LinearRegression.py
├── Student_Performance.csv
├── regression_visualization.png
├── README.md
└── requirements.txt
Installation

Clone the repository:

git clone https://github.com/yourusername/student-performance-linear-regression.git

Install the required packages:

pip install numpy matplotlib

Run the application:

python LinearRegression.py

Before running the project, ensure that the dataset path inside the script points to the correct location of Student_Performance.csv on your system.

Example Machine Learning Pipeline
Student Performance Dataset
            │
            ▼
Data Loading
            │
            ▼
Missing Value Removal
            │
            ▼
Feature Normalization
            │
            ▼
Gradient Descent Training
            │
            ▼
Mean Squared Error Calculation
            │
            ▼
Regression Line Generation
            │
            ▼
Visualization & Model Output
Results

The application successfully learns the relationship between students' previous scores and their overall performance index using Gradient Descent.

The final output includes:

Optimized regression parameters
Reduced Mean Squared Error
Best-fit regression line
Error convergence visualization
Automatically saved graph image

The use of feature normalization allows the model to converge more efficiently while maintaining accurate predictions.

What I Learned

This project strengthened my understanding of how Linear Regression works behind the scenes by implementing every step manually rather than relying on machine learning libraries. It provided practical experience with Gradient Descent optimization, feature normalization, loss functions, and vectorized numerical computation using NumPy.

Developing the model also reinforced the importance of data preprocessing, iterative optimization, and visualization for evaluating model performance. Seeing the error decrease over thousands of training iterations provided valuable insight into how machine learning algorithms gradually improve through optimization.

Future Improvements

Some improvements I'd like to explore include:

Multiple Linear Regression using additional dataset features
Train/Test split evaluation
R² score calculation
Prediction for new user inputs
Model comparison with Scikit-Learn
Interactive prediction interface
Hyperparameter tuning
Support for additional regression algorithms