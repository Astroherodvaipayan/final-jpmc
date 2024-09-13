import numpy as np
import flwr as fl
from flask import Flask, jsonify
import threading

app = Flask(__name__)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return np.round(y_predicted)

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

# Local dataset
X_train = np.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 0, 1],
    [2, 3, 4, 5],
    [6, 7, 8, 9],
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 0, 1, 2]
])
y_train = np.array([0, 0, 1, 1, 1, 1, 1, 1])

# Local model
model = LogisticRegression(learning_rate=0.01, num_iterations=3)

class FederatedClient(fl.client.NumPyClient):
    def __init__(self):
        self.round_weights = []

    def get_parameters(self, config):
        return [model.get_weights(), np.array([model.get_bias()])]

    def set_parameters(self, parameters):
        model.weights = parameters[0]
        model.bias = parameters[1][0]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.fit(X_train, y_train)
        current_weights = self.get_parameters(config)
        self.round_weights.append(current_weights)
        print(f"Round {len(self.round_weights)} weights: {current_weights}")
        return current_weights, len(X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        predictions = model.predict(X_train)
        accuracy = np.mean(predictions == y_train)
        loss = -np.mean(y_train * np.log(predictions) + (1 - y_train) * np.log(1 - predictions))
        return float(loss), len(X_train), {"accuracy": float(accuracy)}

fl_client = FederatedClient()

@app.route('/train3', methods=['POST'])
def train_and_get_weights():
    model.fit(X_train, y_train)
    return jsonify({
        'weights': model.get_weights().tolist(),
        'bias': float(model.get_bias()),
        'predictions': model.predict(X_train).tolist(),
        'all_round_weights': [
            {
                'weights': w[0].tolist(),
                'bias': float(w[1][0])
            } for w in fl_client.round_weights
        ]
    })

def run_flask_app():
    app.run(port=5003)

def print_final_weights():
    print("\n===== Final Weights and Bias =====")
    print(f"Weights: {model.get_weights()}")
    print(f"Bias: {model.get_bias()}")
    print("===================================\n")

if __name__ == '__main__':
    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    # Run Flower client
    fl.client.start_numpy_client(server_address="localhost:8081", client=fl_client)

    # Print final weights after training is complete
    print_final_weights()
