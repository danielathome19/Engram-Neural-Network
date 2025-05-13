import os
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow_engram.models import EngramNetwork


def plot_hebbian_trace(trace_callback, file_path=None):
    # Plot the Hebbian trace evolution over epochs
    plt.figure(figsize=(15, 10))

    # Plot the last hebbian trace as a heatmap
    plt.subplot(2, 2, 1)
    last_trace = trace_callback.traces[-1]  # Get the last trace
    sns.heatmap(last_trace, cmap='viridis')
    plt.title('Final Hebbian Trace')
    plt.xlabel('Hidden Dimension')
    plt.ylabel('Memory Index')

    # Plot the evolution of hebbian trace values
    plt.subplot(2, 2, 2)
    trace_means = [np.mean(np.abs(trace)) for trace in trace_callback.traces]
    trace_maxs = [np.max(trace) for trace in trace_callback.traces]
    trace_mins = [np.min(trace) for trace in trace_callback.traces]
    plt.plot(trace_means, 'b-', label='Mean Abs Value')
    plt.plot(trace_maxs, 'g-', label='Max Value')
    plt.plot(trace_mins, 'r-', label='Min Value')
    plt.title('Hebbian Trace Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    # Plot the first vs last trace for comparison
    if len(trace_callback.traces) > 1:
        plt.subplot(2, 2, 3)
        first_trace = trace_callback.traces[0]
        sns.heatmap(first_trace, cmap='viridis')
        plt.title('Initial Hebbian Trace')
        
        # Plot the difference between first and last
        plt.subplot(2, 2, 4)
        diff = last_trace - first_trace
        sns.heatmap(diff, cmap='coolwarm', center=0)
        plt.title('Trace Difference (Final - Initial)')

    plt.tight_layout()
    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()
    pass


class HebbianTraceMonitor(tf.keras.callbacks.Callback):
    """Callback for monitoring and visualizing Hebbian traces during training.
    
    This callback runs a batch of data through the model after each epoch
    and extracts the Hebbian trace to monitor its evolution over training.
    
    Args:
        sample_batch: A batch of inputs to run through the model for monitoring
        log_dir: Directory to save trace visualizations (if None, no saving)
        save_freq: Save visualizations every N epochs (default: 5)
        verbose: Verbosity level (0=silent, 1=basic stats, 2=detailed)
        plot_every: Generate plots every N epochs (default: 5)
    """
    
    def __init__(self, 
                 sample_batch, 
                 log_dir=None, 
                 save_freq=5,
                 verbose=1, 
                 plot_every=5):
        super().__init__()
        self.sample_batch = sample_batch
        self.log_dir = log_dir
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.save_freq = save_freq
        self.verbose = verbose
        self.plot_every = plot_every
        self.traces = []
        self.attention_weights = []
        self.stats = {
            'mean_abs': [],
            'max': [],
            'min': [],
            'std': [],
            'sparsity': []
        }
        
    def on_train_begin(self, logs=None):
        """Initialize visualization on training start."""
        if self.verbose > 0:
            print("HebbianTraceMonitor initialized.")
            # print("Will monitor Hebbian trace evolution over training.")
            
    def _find_hebbian_cell(self):
        """Find the EngramCell in the model."""
        # Look through all layers to find the RNN with a EngramCell
        for layer in self.model.layers:
            if isinstance(layer, EngramNetwork):
                if hasattr(layer.cell, 'hebbian_lr'):  # Identify by unique attribute
                    return layer.cell
            elif hasattr(layer, 'engram_cell'):
                if hasattr(layer.engram_cell, 'hebbian_lr'):
                    return layer.engram_cell
                    
        # If we're using a custom model with a direct cell reference
        if hasattr(self.model, 'cell') and hasattr(self.model.cell, 'hebbian_lr'):
            return self.model.cell
            
        raise ValueError("Could not find EngramCell in the model")
        
    def _find_rnn_layer(self):
        """Find the RNN layer that contains the EngramCell."""
        for layer in self.model.layers:
            # if isinstance(layer, tf.keras.layers.RNN):
            #     if hasattr(layer.cell, 'hebbian_lr'):
            #         return layer
            if isinstance(layer, EngramNetwork):
                return layer.rnn
            elif hasattr(layer, 'engram_network'):
                return layer.engram_network.rnn
                    
        raise ValueError("Could not find RNN layer with EngramCell")
            
    def on_epoch_end(self, epoch, logs=None):
        """Extract and store Hebbian trace at the end of each epoch."""
        # Find the cell and RNN layer
        cell = self._find_hebbian_cell()
        rnn_layer = self._find_rnn_layer()
        
        # Get a small batch for monitoring (limit to 64 samples for efficiency)
        if len(self.sample_batch) > 64:
            monitoring_batch = self.sample_batch[:64]
        else:
            monitoring_batch = self.sample_batch
            
        # Initialize states
        initial_state = cell.get_initial_state(
            batch_size=tf.shape(monitoring_batch)[0], 
            dtype=tf.float32
        )
        
        # Run the data through the RNN layer to get states
        # Note: Using return_state=True for the RNN layer is important
        outputs = rnn_layer(monitoring_batch, 
                            initial_state=initial_state, 
                            training=False)
                            # return_state=True)
        
        # Extract state information
        if isinstance(outputs, (list, tuple)):
            # First element is the output, remaining elements are states
            _, *states = outputs
            hebbian_trace = states[1]  # Second state is the hebbian trace

            # Store the results
            self.traces.append(hebbian_trace.numpy())
            
            # Compute statistics
            trace_np = hebbian_trace.numpy()
            self.stats['mean_abs'].append(np.mean(np.abs(trace_np)))
            self.stats['max'].append(np.max(trace_np))
            self.stats['min'].append(np.min(trace_np))
            self.stats['std'].append(np.std(trace_np))
            
            # Calculate sparsity as percentage of values close to zero
            sparsity = np.mean(np.abs(trace_np) < 0.01)
            self.stats['sparsity'].append(sparsity)
            
            # Print statistics if requested
            if self.verbose > 0:
                print(f"\nHebbian trace [Epoch {epoch+1}] - "
                      f"min: {self.stats['min'][-1]:.4f}, "
                      f"max: {self.stats['max'][-1]:.4f}, "
                      f"mean abs: {self.stats['mean_abs'][-1]:.4f}, "
                      f"sparsity: {self.stats['sparsity'][-1]:.2%}")
                      
            # Generate plots periodically
            if epoch % self.plot_every == 0:
                self.plot_trace_evolution(epoch)
                
            # Save plots to disk if requested
            if self.log_dir and epoch % self.save_freq == 0:
                self.save_trace_plots(epoch)
                
    def plot_trace_evolution(self, epoch=None):
        """Plot the evolution of the Hebbian trace."""
        if not self.traces:
            return
            
        # Create a 2x2 grid of subplots with explicit indexing
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot the current trace as a heatmap (top-left)
        current_trace = self.traces[-1]
        sns.heatmap(current_trace, cmap='viridis', ax=axs[0, 0])
        axs[0, 0].set_title(f'Current Hebbian Trace (Epoch {epoch+1 if epoch is not None else len(self.traces)})')
        axs[0, 0].set_xlabel('Hidden Dimension')
        axs[0, 0].set_ylabel('Memory Index')
        
        # Plot the evolution of trace statistics (top-right)
        epochs = range(1, len(self.stats['mean_abs'])+1)
        axs[0, 1].plot(epochs, self.stats['mean_abs'], 'b-', label='Mean Abs Value')
        axs[0, 1].plot(epochs, self.stats['max'], 'g-', label='Max Value')
        axs[0, 1].plot(epochs, self.stats['min'], 'r-', label='Min Value')
        axs[0, 1].plot(epochs, self.stats['sparsity'], 'c--', label='Sparsity')
        axs[0, 1].set_title('Hebbian Trace Evolution')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Value')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
        
        # Plot the first trace if we have more than one (bottom-left)
        if len(self.traces) > 1:
            first_trace = self.traces[0]
            sns.heatmap(first_trace, cmap='viridis', ax=axs[1, 0])
            axs[1, 0].set_title('Initial Hebbian Trace')
            axs[1, 0].set_xlabel('Hidden Dimension')
            axs[1, 0].set_ylabel('Memory Index')
            
            # Plot the difference between first and current trace (bottom-right)
            diff = current_trace - first_trace
            sns.heatmap(diff, cmap='coolwarm', center=0, ax=axs[1, 1])
            axs[1, 1].set_title('Trace Difference (Current - Initial)')
            axs[1, 1].set_xlabel('Hidden Dimension')
            axs[1, 1].set_ylabel('Memory Index')
            
        plt.tight_layout()
        plt.show()
        
    def save_trace_plots(self, epoch):
        """Save visualization to disk."""
        if not self.log_dir:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.log_dir, f"hebbian_trace_epoch{epoch+1}_{timestamp}.png")
        
        # Create a 2x2 grid of subplots with explicit indexing
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot the current trace as a heatmap (top-left)
        current_trace = self.traces[-1]
        sns.heatmap(current_trace, cmap='viridis', ax=axs[0, 0])
        axs[0, 0].set_title(f'Hebbian Trace (Epoch {epoch+1})')
        axs[0, 0].set_xlabel('Hidden Dimension')
        axs[0, 0].set_ylabel('Memory Index')
        
        # Plot the evolution of trace statistics (top-right)
        epochs = range(1, len(self.stats['mean_abs'])+1)
        axs[0, 1].plot(epochs, self.stats['mean_abs'], 'b-', label='Mean Abs Value')
        axs[0, 1].plot(epochs, self.stats['max'], 'g-', label='Max Value')
        axs[0, 1].plot(epochs, self.stats['min'], 'r-', label='Min Value')
        axs[0, 1].plot(epochs, self.stats['sparsity'], 'c--', label='Sparsity')
        axs[0, 1].set_title('Hebbian Trace Evolution')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Value')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
        
        # Plot the first trace if we have more than one (bottom-left)
        if len(self.traces) > 1:
            first_trace = self.traces[0]
            sns.heatmap(first_trace, cmap='viridis', ax=axs[1, 0])
            axs[1, 0].set_title('Initial Hebbian Trace')
            axs[1, 0].set_xlabel('Hidden Dimension')
            axs[1, 0].set_ylabel('Memory Index')
            
            # Plot the difference (bottom-right)
            diff = current_trace - first_trace
            sns.heatmap(diff, cmap='coolwarm', center=0, ax=axs[1, 1])
            axs[1, 1].set_title('Trace Difference (Current - Initial)')
            axs[1, 1].set_xlabel('Hidden Dimension')
            axs[1, 1].set_ylabel('Memory Index')
            
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        
        if self.verbose > 0:
            print(f"Saved Hebbian trace visualization to {filename}")
            
    def get_trace_evolution(self):
        """Return the complete history of Hebbian traces for analysis."""
        return self.traces
        
    def get_trace_stats(self):
        """Return statistics about Hebbian trace evolution."""
        return self.stats