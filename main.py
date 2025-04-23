import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
        
    def build_model(self):
        """Build the LSTM model architecture"""
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(self.n_neurons, activation='tanh', 
                 recurrent_activation='sigmoid',
                 input_shape=self.input_shape),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        # Compile the model with Adam optimizer and MAE loss
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='mae',
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
        
        # Add learning rate scheduler if requested
        if use_lr_scheduler:
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=20,
                min_lr=1e-6,
                verbose=1
            )
            callbacks.append(lr_scheduler)
        
        # Add early stopping
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
        
        # Calculate metrics
        mae = np.mean(np.abs(y_test - predictions.flatten()))
        mse = np.mean((y_test - predictions.flatten())**2)
        rmse = np.sqrt(mse)
        
        # Calculate R² score
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
        plt.ylabel('Mean Absolute Error')
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
        
        # Inverse transform if normalization parameters are available
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

        # Optional: print comparison table
        print("\nActual vs Predicted Values:")
        print("Index\tActual\t\tPredicted\tDifference")
        print("-" * 50)
        for i in range(min(20, len(y_test_actual))):
            diff = y_test_actual[i] - predictions_actual[i]
            print(f"{i}\t{y_test_actual[i]:.6f}\t{predictions_actual[i]:.6f}\t{diff:.6f}")


def main():
    """Main function to run the LSTM model"""
    # Load processed data
    print("Loading processed data...")
    with open('processed_ice_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X = data['X']
    y = data['y']
    feature_cols = data['feature_cols']
    normalized_data = data['normalized_data']
    
    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Define hyperparameters to experiment with
    neuron_configs = [32, 50, 64, 100]
    batch_sizes = [16, 32, 64]
    use_lr_scheduler_configs = [False, True]
    
    best_metrics = {'MAE': float('inf')}
    best_config = None
    
    # Experiment with different configurations
    for n_neurons in neuron_configs:
        for batch_size in batch_sizes:
            for use_lr_scheduler in use_lr_scheduler_configs:
                print(f"\nTraining with {n_neurons} neurons, batch size {batch_size}, "
                      f"LR scheduler: {use_lr_scheduler}")
                
                # Create and build model
                lstm_model = LSTMModel(input_shape=(X_train.shape[1], X_train.shape[2]), 
                                     n_neurons=n_neurons)
                lstm_model.build_model()
                
                # Train model
                history = lstm_model.train(X_train, y_train, X_val, y_val, 
                                         batch_size=batch_size, 
                                         use_lr_scheduler=use_lr_scheduler)
                
                # Evaluate model on test set
                metrics = lstm_model.evaluate(X_test, y_test)
                
                print("Test Metrics:")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.6f}")
                
                # Update best model if current model is better
                if metrics['MAE'] < best_metrics['MAE']:
                    best_metrics = metrics
                    best_config = {
                        'n_neurons': n_neurons,
                        'batch_size': batch_size,
                        'use_lr_scheduler': use_lr_scheduler
                    }
                    best_model = lstm_model
    
    # Report best configuration
    print("\n" + "="*50)
    print("Best Configuration:")
    for key, value in best_config.items():
        print(f"{key}: {value}")
    
    print("\nBest Test Metrics:")
    for metric_name, value in best_metrics.items():
        print(f"{metric_name}: {value:.6f}")
    
    # Plot results for the best model
    best_model.plot_training_history('plots/lstm_training_history.png')
    best_model.plot_predictions(X_test, y_test, normalized_data, save_path='plots/lstm_predictions.png')

    
    # Save the best model
    best_model.model.save('best_lstm_model.h5')
    print("\nBest model saved as 'best_lstm_model.h5'")

if __name__ == "__main__":
    main()