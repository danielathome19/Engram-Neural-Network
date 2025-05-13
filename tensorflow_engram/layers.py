import tensorflow as tf
import tensorflow_engram
from tensorflow.keras import layers

class EngramCell(tf.keras.layers.Layer):
    """Biologically-inspired RNN cell with Hebbian learning and memory trace.
    
    This cell implements engram-like memory formation using both gradient-based
    learning (for memory banks) and Hebbian plasticity (for memory traces).
    
    Args:
        hidden_dim: Dimensionality of the hidden state
        memory_size: Number of memory slots in the memory bank
        hebbian_lr: Learning rate for Hebbian trace updates
        scale_factor: Weight given to Hebbian trace vs memory bank (0-1)
        temperature: Temperature for softmax attention (higher = softer attention)
        use_layer_norm: Whether to use layer normalization
        sparsity_strength: Strength of (L1) sparsity regularization on attention weights (0-1)
    """
    def __init__(self, 
                 hidden_dim, 
                 memory_size, 
                 hebbian_lr=0.05,
                 scale_factor=0.5,
                 temperature=1.5, 
                 use_layer_norm=True,
                 sparsity_strength=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.hebbian_lr = hebbian_lr
        self.scale_factor = scale_factor
        self.temperature = temperature
        self.use_layer_norm = use_layer_norm
        self.sparsity_strength = sparsity_strength
        
    @property
    def state_size(self):
        # Properly define state sizes for TensorFlow RNN infrastructure
        return [
            tf.TensorShape([self.hidden_dim]), 
            tf.TensorShape([self.memory_size, self.hidden_dim]), 
            tf.TensorShape([self.memory_size, self.hidden_dim])
        ]
    
    @property
    def output_size(self):
        return self.hidden_dim
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Input encoder with ReLU activation for better gradient flow
        self.input_encoder = layers.Dense(
            self.hidden_dim, 
            activation="relu",
            kernel_initializer="he_normal"
        )
        
        # Hidden state integrator
        self.hidden_integrator = layers.Dense(
            self.hidden_dim * 3, 
            activation="relu",
            kernel_initializer="he_normal"
        )
        
        # Output transformer
        self.output_dense = layers.Dense(
            self.hidden_dim, 
            activation="relu",
            kernel_initializer="he_normal"
        )
        
        # Output decoder
        self.decoder = layers.Dense(self.hidden_dim)
        
        # Memory Bank (Engrams) - learnable parameters
        self.memory_bank = self.add_weight(
            shape=(self.memory_size, self.hidden_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="engram_memory"
        )
        
        # Initial hebbian trace for state setup
        self.initial_hebbian_trace = self.add_weight(
            shape=(self.memory_size, self.hidden_dim),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01),
            trainable=False,
            name="initial_hebbian_trace"
        )
        
        # Layer normalization for stability
        if self.use_layer_norm:
            self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.built = True
    
    def call(self, inputs, states, training=None):
        # Unpack states in standard AbstractRNNCell format
        prev_h, hebbian_trace, memory_bank_state = states
        
        # Encode input with normalization
        z_t = self.input_encoder(inputs)
        if self.use_layer_norm:
            z_t = self.layer_norm1(z_t)
        
        # Combine memory bank and hebbian trace
        effective_memory = memory_bank_state + self.scale_factor * hebbian_trace
        
        # Retrieve engram via cosine similarity
        z_norm = tf.nn.l2_normalize(z_t, axis=-1)
        mem_norm = tf.nn.l2_normalize(effective_memory, axis=-1)
        
        # Compute attention with temperature scaling
        # Lower sparsity temperature = sharper distribution
        sim = tf.matmul(z_norm, mem_norm, transpose_b=True)
        sparsity_temperature = self.temperature / (1.0 + self.sparsity_strength * 10.0)
        attention_weights = tf.nn.softmax(sim / sparsity_temperature, axis=-1)

        # Retrieved memory
        m_t = tf.matmul(attention_weights, effective_memory)
        
        # Integrate memory with previous hidden state and current input
        combined = tf.concat([z_t, m_t, prev_h], axis=-1)
        combined_hidden = self.hidden_integrator(combined)
        h_t = self.output_dense(combined_hidden)
        
        if self.use_layer_norm:
            h_t = self.layer_norm2(h_t)
        
        # Apply Hebbian learning: correlation between attention and current state
        hebbian_update = tf.einsum('bi,bj->bij', attention_weights, z_t)
        mean_update = tf.reduce_mean(hebbian_update, axis=0)
        
        # Add small gradient noise for exploration
        noise = tf.random.normal(
            shape=mean_update.shape, 
            mean=0.0, 
            stddev=0.001,
            dtype=mean_update.dtype
        )
        
        # Update Hebbian trace with decay term
        decay_rate = 1.0 - self.hebbian_lr
        new_hebbian_trace = hebbian_trace * decay_rate + self.hebbian_lr * (mean_update + noise)
        
        # Clip to prevent explosion
        new_hebbian_trace = tf.clip_by_value(new_hebbian_trace, -0.1, 0.1)
        
        # Return outputs and state in AbstractRNNCell format
        new_state = [h_t, new_hebbian_trace, memory_bank_state]
        return h_t, new_state
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None and inputs is not None:
            batch_size = tf.shape(inputs)[0]
        
        # Initial hidden state - small random values
        initial_h = tf.random.normal(
            [batch_size, self.hidden_dim], 
            mean=0.0, 
            stddev=0.01, 
            dtype=dtype or tf.float32
        )
        
        # Initial hebbian trace - small random values
        initial_hebbian = tf.random.normal(
            [self.memory_size, self.hidden_dim], 
            mean=0.0, 
            stddev=0.01, 
            dtype=dtype or tf.float32
        )
        
        # Initial memory bank state is the learned memory bank weights
        initial_memory_bank = self.memory_bank
        
        return [initial_h, initial_hebbian, initial_memory_bank]
    
    def get_config(self):
        config = {
            'hidden_dim': self.hidden_dim,
            'memory_size': self.memory_size,
            'hebbian_lr': self.hebbian_lr,
            'scale_factor': self.scale_factor,
            'temperature': self.temperature,
            'use_layer_norm': self.use_layer_norm,
            'sparsity_strength': self.sparsity_strength
        }
        base_config = super().get_config()
        return {**base_config, **config}


class EngramAttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(EngramAttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.attention_dense = layers.Dense(1, activation='tanh')
        super(EngramAttentionLayer, self).build(input_shape)
        
    def call(self, inputs):
        # Compute attention weights
        attention_weights = self.attention_dense(inputs)  # [batch, time_steps, 1]
        attention_weights = tf.keras.activations.softmax(attention_weights, axis=1)
        
        # Apply attention weights to input
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector


class Engram(layers.Layer):
    def __init__(self, engram_network, return_states=False, **kwargs):
        super().__init__(**kwargs)
        self.engram_network = engram_network
        self.engram_cell = engram_network.cell
        self.wrapper_return_states = return_states
        
        # Store network parameters for serialization
        self.hidden_dim = engram_network.hidden_dim
        self.memory_size = engram_network.memory_size
        self.hebbian_lr = engram_network.hebbian_lr
        self.return_sequences = engram_network.return_sequences
        self.return_states = engram_network.return_states
        self.reset_states_per_batch = engram_network.reset_states_per_batch
        self.stateful = engram_network.stateful
        self.use_attention = engram_network.use_attention
        self.dropout_rate = engram_network.dropout_rate

    def call(self, inputs, *args, **kwargs):
        outputs = self.engram_network(inputs, *args, **kwargs)
        if self.wrapper_return_states:
            return outputs[0]  # Extract only the main output
        return outputs

    def build(self, input_shape):
        # Call the build method of the underlying engram_network
        self.engram_network.build(input_shape)
        super().build(input_shape)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "memory_size": self.memory_size,
            "hebbian_lr": self.hebbian_lr,
            "return_sequences": self.return_sequences,
            "return_states": self.return_states,
            "reset_states_per_batch": self.reset_states_per_batch,
            "stateful": self.stateful,
            "use_attention": self.use_attention,
            "dropout_rate": self.dropout_rate,
            "wrapper_return_states": self.wrapper_return_states,
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        # Extract network parameters
        hidden_dim = config.pop("hidden_dim")
        memory_size = config.pop("memory_size")
        hebbian_lr = config.pop("hebbian_lr")
        return_sequences = config.pop("return_sequences")
        return_states = config.pop("return_states")
        reset_states_per_batch = config.pop("reset_states_per_batch")
        stateful = config.pop("stateful")
        use_attention = config.pop("use_attention")
        dropout_rate = config.pop("dropout_rate")
        wrapper_return_states = config.pop("wrapper_return_states")
        
        # Create a new EngramNetwork
        engram_network = tensorflow_engram.models.EngramNetwork(
            hidden_dim=hidden_dim,
            memory_size=memory_size,
            hebbian_lr=hebbian_lr,
            return_sequences=return_sequences,
            return_states=return_states,
            reset_states_per_batch=reset_states_per_batch,
            stateful=stateful,
            use_attention=use_attention,
            dropout_rate=dropout_rate
        )
        
        # Create a new wrapper
        return cls(engram_network, wrapper_return_states, **config)
