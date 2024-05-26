import numpy as np

# دالة لحساب دالة التكلفة
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    sq_errors = (predictions - y) ** 2
    return 1 / (2 * m) * np.sum(sq_errors)

# دالة لتنفيذ خوارزمية Gradient Descent
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = np.zeros(num_iters)
    
    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        delta = (1 / m) * (X.T.dot(errors))
        theta = theta - alpha * delta
        cost_history[i] = compute_cost(X, y, theta)
    
    return theta, cost_history

# إعداد البيانات
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
y = np.array([6, 5, 7, 10])
theta = np.zeros(2)
alpha = 0.01
num_iters = 1000

# تنفيذ خوارزمية Gradient Descent
theta, cost_history = gradient_descent(X, y, theta, alpha, num_iters)

# عرض النتائج
print("Theta:", theta)
print("Final cost:", cost_history[-1])￼Enter
