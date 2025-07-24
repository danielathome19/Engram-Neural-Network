import tensorflow as tf
from keras import layers, Model, Sequential
from tensorflow_engram.layers import EngramCell, EngramAttentionLayer, Engram


class EngramNetwork(Model):
    """Ready-to-use Hebbian Engram Network for sequence modeling tasks.
    
    This model wraps the HebbianEngramCell with common configurations for
    sequence processing, adding appropriate input/output layers.
    """
    
    def __init__(self, 
                 hidden_dim=128,
                 memory_size=64,
                 hebbian_lr=0.05,
                 output_units=None,
                 output_activation=None,
                 return_sequences=True,
                 return_states=False,
                 reset_states_per_batch=True,
                 stateful=False,
                 use_attention=False,
                 dropout_rate=0.0,
                 sparsity_strength=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.hebbian_lr = hebbian_lr
        self.return_sequences = return_sequences
        self.return_states = return_states
        self.stateful = stateful
        self.output_units = output_units
        self.output_activation = output_activation
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        self.reset_states_per_batch = reset_states_per_batch
        self._batch_counter = tf.Variable(0, dtype=tf.int64, trainable=False)
        
        # Create the cell and RNN layer
        self.cell = EngramCell(
            hidden_dim=hidden_dim,
            memory_size=memory_size,
            hebbian_lr=hebbian_lr,
            sparsity_strength=sparsity_strength
        )
        
        self.rnn = layers.RNN(
            self.cell,
            return_sequences=return_sequences,
            return_state=return_states,
            stateful=stateful
        )
        
        if self.use_attention:
            self.attention = EngramAttentionLayer()
            
        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(dropout_rate)
            
        # Output layer if specified
        if self.output_units is not None:
            self.output_layer = layers.Dense(
                output_units, 
                activation=output_activation
            )
            
    def build(self, input_shape):
        # RNN will be built when called
        super().build(input_shape)
        
    def call(self, inputs, training=None, initial_state=None):
        # Reset states logic for training mode
        if training and self.reset_states_per_batch:
            self._batch_counter.assign_add(1)
            # Force new initial state
            batch_size = tf.shape(inputs)[0]
            initial_state = self.cell.get_initial_state(batch_size=batch_size, dtype=inputs.dtype)
        
        outputs = self.rnn(inputs, training=training, initial_state=initial_state)
        
        # Handle different output configurations
        if self.return_states:
            rnn_output, *states = outputs
        else:
            rnn_output = outputs
            
        if self.use_attention and self.return_sequences:
            rnn_output = self.attention(rnn_output)
            
        if self.dropout_rate > 0:
            rnn_output = self.dropout(rnn_output, training=training)
            
        if self.output_units is not None:
            if self.return_sequences and not self.use_attention:
                rnn_output = layers.TimeDistributed(self.output_layer)(rnn_output)
            else:
                rnn_output = self.output_layer(rnn_output)
                
        if self.return_states:
            return [rnn_output] + states
        else:
            return rnn_output
            
    def get_config(self):
        config = {
            'hidden_dim': self.hidden_dim,
            'memory_size': self.memory_size,
            'hebbian_lr': self.hebbian_lr,
            'output_units': self.output_units,
            'output_activation': self.output_activation,
            'return_sequences': self.return_sequences,
            'return_states': self.return_states,
            'stateful': self.stateful,
            'use_attention': self.use_attention,
            'dropout_rate': self.dropout_rate,
        }
        base_config = super().get_config()
        return {**base_config, **config}


def engram_classifier(input_shape, num_classes, hidden_dim=128, memory_size=64, return_states=False, reset_states_per_batch=True,  return_sequences=False, use_attention=True, sparsity_strength=0.1, **kwargs):
    """Creates a classification model using Engram Network.
    
    Args:
        input_shape: Shape of input sequences (timesteps, features)
        num_classes: Number of output classes
        hidden_dim: Dimension of hidden states
        memory_size: Size of memory bank
        return_states: Whether to return states from the RNN
        reset_states_per_batch: Whether to reset states at the beginning of each batch
        sparsity_strength: Strength of (L1) sparsity regularization on attention weights (0-1)
        **kwargs: Additional arguments to pass to EngramNetwork
        
    Returns:
        A compiled Keras model ready for training
    """
    engram_network = EngramNetwork(
        hidden_dim=hidden_dim,
        memory_size=memory_size,
        return_sequences=return_sequences,
        use_attention=use_attention,
        return_states=return_states,
        reset_states_per_batch=reset_states_per_batch,
        sparsity_strength=sparsity_strength,
        **kwargs
    )
    
    # Wrap the EngramNetwork to handle multiple outputs if return_states=True
    model = Sequential([
        layers.InputLayer(input_shape=input_shape),
        Engram(engram_network, return_states=return_states),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax') if not return_sequences else \
        layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))
    ])

    return model


def engram_regression(input_shape, output_dim, hidden_dim=128, memory_size=64, return_states=False, reset_states_per_batch=True, sparsity_strength=0.1, **kwargs):
    """Creates a regression model using Engram Network.
    
    Args:
        input_shape: Shape of input sequences (timesteps, features)
        output_dim: Dimension of regression outputs
        hidden_dim: Dimension of hidden states
        memory_size: Size of memory bank
        return_states: Whether to return the states of the RNN
        reset_states_per_batch: Whether to reset states at the beginning of each batch
        sparsity_strength: Strength of (L1) sparsity regularization on attention weights (0-1)
        **kwargs: Additional arguments to pass to EngramNetwork
        
    Returns:
        A compiled Keras model ready for training
    """
    model = Sequential([
        layers.InputLayer(input_shape=input_shape),
        EngramNetwork(
            hidden_dim=hidden_dim,
            memory_size=memory_size,
            return_sequences=True,
            return_states=return_states,
            reset_states_per_batch=reset_states_per_batch,
            sparsity_strength=sparsity_strength,
            **kwargs
        ),
        layers.TimeDistributed(layers.Dense(output_dim))
    ])
    
    return model
