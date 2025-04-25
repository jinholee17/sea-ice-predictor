import tensorflow as tf
import numpy as np
import pickle
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

class LSTMModel:
    def __init__(self, input_shape, n_neurons=50):
        """
        Initialize LSTM model for sea ice extent prediction
        
        Args:
        - input_shape: Tuple of (sequence_length, num_features)
        - n_neurons: Number of LSTM neurons
        """
        self.input_shape = input_shape
        self.n_neurons = n_neurons
        self.model = None
        self.history = None
        self.learning_rate = None
        self.loss_function = None
        
    def build_model(self, learning_rate=0.001, loss_function='mae'):
        """
        Build the LSTM model architecture
        
        Args:
        - learning_rate: Learning rate for Adam optimizer
        - loss_function: Loss function to use ('mae' or 'mse')
        """
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(self.n_neurons, activation='tanh', 
                 recurrent_activation='sigmoid',
                 input_shape=self.input_shape),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_function,
            metrics=['mse', 'mae']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=200, batch_size=32, use_lr_scheduler=False):
        """
        Train the LSTM model
        
        Args:
        - X_train, y_train: Training data
        - X_val, y_val: Validation data
        - epochs: Number of training epochs
        - batch_size: Training batch size
        - use_lr_scheduler: Whether to use learning rate scheduling
        
        Returns:
        - History object from training
        """
        callbacks = []
        
        if use_lr_scheduler:
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=20,
                min_lr=1e-6,
                verbose=1
            )
            callbacks.append(lr_scheduler)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
        - X_test, y_test: Test data
        
        Returns:
        - Dictionary of metrics
        """
        predictions = self.model.predict(X_test)
        
        mae = np.mean(np.abs(y_test - predictions.flatten()))
        mse = np.mean((y_test - predictions.flatten())**2)
        rmse = np.sqrt(mse)
        
        ss_res = np.sum((y_test - predictions.flatten())**2)
        ss_tot = np.sum((y_test - np.mean(y_test))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
        
        return metrics
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot and save training and validation loss curves"""
        if self.history is None:
            print("No training history available")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel(f'{"Mean Absolute Error" if self.loss_function == "mae" else "Mean Squared Error"}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
        plt.close()

    
    def plot_predictions(self, X_test, y_test, normalized_data, target='extent', save_path='predictions.png'):
        """
        Plot and save actual vs predicted values
        """
        predictions = self.model.predict(X_test)
        
        if f'{target}_min' in normalized_data and f'{target}_max' in normalized_data:
            min_val = normalized_data[f'{target}_min']
            max_val = normalized_data[f'{target}_max']
            y_test_actual = y_test * (max_val - min_val) + min_val
            predictions_actual = predictions.flatten() * (max_val - min_val) + min_val
        else:
            y_test_actual = y_test
            predictions_actual = predictions.flatten()
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_actual, label='Actual', marker='o')
        plt.plot(predictions_actual, label='Predicted', marker='x')
        plt.title('Actual vs Predicted Sea Ice Extent')
        plt.xlabel('Time Steps')
        plt.ylabel('Sea Ice Extent')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Prediction plot saved to {save_path}")
        plt.close()

        print("\nActual vs Predicted Values:")
        print("Index\tActual\t\tPredicted\tDifference")
        print("-" * 50)
        for i in range(min(20, len(y_test_actual))):
            diff = y_test_actual[i] - predictions_actual[i]
            print(f"{i}\t{y_test_actual[i]:.6f}\t{predictions_actual[i]:.6f}\t{diff:.6f}")


class ExperimentTracker:
    """Class to track experiments and create comparison tables"""
    
    def __init__(self):
        self.experiments = []
        self.best_experiment = None
        self.best_metrics = {'MAE': float('inf')}
    
    def add_experiment(self, experiment_config, metrics, model=None):
        """
        Add experiment results to tracker
        
        Args:
        - experiment_config: Dictionary of experiment configuration
        - metrics: Dictionary of evaluation metrics
        - model: The trained model (optional)
        """
        experiment = {
            'config': experiment_config,
            'metrics': metrics,
            'model': model
        }
        
        self.experiments.append(experiment)
        
        if metrics['MAE'] < self.best_metrics['MAE']:
            self.best_metrics = metrics
            self.best_experiment = experiment
    
    def print_results_table(self):
        """Print formatted table of experiment results"""
        sorted_experiments = sorted(self.experiments, key=lambda x: x['metrics']['MAE'])
        
        print("\n" + "="*100)
        print("LSTM MODEL EXPERIMENTS RESULTS")
        print("="*100)
        headers = "| # | Neurons | Learning Rate | Loss Func | Batch Size | LR Scheduler | MAE | MSE | RMSE | R² |"
        print(headers)
        print("|" + "-"*(len(headers)-2) + "|")
        
        for idx, exp in enumerate(sorted_experiments):
            config = exp['config']
            metrics = exp['metrics']
            
            print(
                f"| {idx+1} | {config['n_neurons']} | {config['learning_rate']:.6f} | "
                f"{config['loss_function']} | {config['batch_size']} | "
                f"{str(config['use_lr_scheduler']):5} | {metrics['MAE']:.6f} | "
                f"{metrics['MSE']:.6f} | {metrics['RMSE']:.6f} | {metrics['R2']:.6f} |"
            )
        
        print("="*100)
        best_idx = self.experiments.index(self.best_experiment)
        print(f"Best model: #{best_idx+1} (MAE: {self.best_metrics['MAE']:.6f})")
        print("="*100)
    
    def save_results_to_csv(self, filename='lstm_experiments.csv'):
        """Save experiment results to CSV file"""
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'Experiment', 'Neurons', 'Learning Rate', 'Loss Function', 
                'Batch Size', 'LR Scheduler', 'MAE', 'MSE', 'RMSE', 'R²'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for idx, exp in enumerate(self.experiments):
                config = exp['config']
                metrics = exp['metrics']
                
                writer.writerow({
                    'Experiment': idx+1,
                    'Neurons': config['n_neurons'],
                    'Learning Rate': config['learning_rate'],
                    'Loss Function': config['loss_function'],
                    'Batch Size': config['batch_size'],
                    'LR Scheduler': config['use_lr_scheduler'],
                    'MAE': metrics['MAE'],
                    'MSE': metrics['MSE'],
                    'RMSE': metrics['RMSE'],
                    'R²': metrics['R2']
                })
        
        print(f"Experiment results saved to {filename}")
    
    def get_best_model(self):
        """Return the best model based on MAE"""
        if self.best_experiment:
            return self.best_experiment['model']
        return None


def main():
    """Main function to run the LSTM model experiments"""
    print("Loading processed data...")
    with open('processed_ice_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X = data['X']
    y = data['y']
    feature_cols = data['feature_cols']
    normalized_data = data['normalized_data']
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    tracker = ExperimentTracker()
    
    neuron_configs = [32, 50, 64, 100]
    learning_rates = [0.001, 0.01, 0.0001]
    loss_functions = ['mae', 'mse']
    batch_sizes = [16, 32, 64]
    use_lr_scheduler_configs = [False, True]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for n_neurons in neuron_configs:
        for learning_rate in learning_rates:
            for loss_function in loss_functions:
                for batch_size in batch_sizes:
                    for use_lr_scheduler in use_lr_scheduler_configs:
                        print(f"\nTraining with {n_neurons} neurons, learning rate {learning_rate}, "
                              f"loss function {loss_function}, batch size {batch_size}, "
                              f"LR scheduler: {use_lr_scheduler}")
                        
                        lstm_model = LSTMModel(input_shape=(X_train.shape[1], X_train.shape[2]), 
                                             n_neurons=n_neurons)
                        lstm_model.build_model(learning_rate=learning_rate, loss_function=loss_function)
                        
                        history = lstm_model.train(X_train, y_train, X_val, y_val, 
                                                 batch_size=batch_size, 
                                                 use_lr_scheduler=use_lr_scheduler)
                        
                        metrics = lstm_model.evaluate(X_test, y_test)
                        
                        print("Test Metrics:")
                        for metric_name, value in metrics.items():
                            print(f"{metric_name}: {value:.6f}")
                        
                        experiment_config = {
                            'n_neurons': n_neurons,
                            'learning_rate': learning_rate,
                            'loss_function': loss_function,
                            'batch_size': batch_size,
                            'use_lr_scheduler': use_lr_scheduler
                        }
                        tracker.add_experiment(experiment_config, metrics, lstm_model)
    
    tracker.print_results_table()
    
    csv_filename = f"lstm_experiments_{timestamp}.csv"
    tracker.save_results_to_csv(csv_filename)
    
    best_model = tracker.get_best_model()
    best_config = tracker.best_experiment['config']
    
    best_model.plot_training_history(f'plots/lstm_training_history_{timestamp}.png')
    best_model.plot_predictions(X_test, y_test, normalized_data, save_path=f'plots/lstm_predictions_{timestamp}.png')
    
    model_filename = f"best_lstm_model_{timestamp}.h5"
    best_model.model.save(model_filename)
    print(f"\nBest model saved as '{model_filename}'")
    
    print("\nCreating experiment summary report...")
    summary_filename = f"experiment_summary_{timestamp}.txt"
    with open(summary_filename, 'w') as f:
        f.write("="*50 + "\n")
        f.write("LSTM MODEL EXPERIMENT SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write("Best Configuration:\n")
        for key, value in best_config.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nBest Metrics:\n")
        for metric_name, value in tracker.best_metrics.items():
            f.write(f"{metric_name}: {value:.6f}\n")
        
        f.write("\nRecommendation:\n")
        f.write(f"Based on the experiments, the recommended configuration uses {best_config['n_neurons']} neurons ")
        f.write(f"with a learning rate of {best_config['learning_rate']}, {best_config['loss_function']} loss function, ")
        f.write(f"batch size of {best_config['batch_size']}, and ")
        f.write(f"{'uses' if best_config['use_lr_scheduler'] else 'does not use'} learning rate scheduling.\n\n")
        
        f.write("This configuration achieved the following performance metrics on the test set:\n")
        for metric_name, value in tracker.best_metrics.items():
            f.write(f"- {metric_name}: {value:.6f}\n")
    
    print(f"Summary report saved as '{summary_filename}'")


if __name__ == "__main__":
    main()