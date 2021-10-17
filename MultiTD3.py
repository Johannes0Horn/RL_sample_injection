# Library Imports
import numpy as np
import tensorflow as tf


class Critic(tf.keras.Model):
    """Defines a Critic Deep Learning Network"""

    def __init__(self, dim_actions, H1_dim=512, H2_dim=512, name='critic'):
        super(Critic, self).__init__()
        self.H1_dim = H1_dim
        self.H2_dim = H2_dim
        self.dim_actions = dim_actions
        self.model_name = name
        self.checkpoint = self.model_name + '.h5'
        self.H1 = tf.keras.layers.Dense(self.H1_dim, activation='relu')
        self.H2 = tf.keras.layers.Dense(self.H2_dim, activation='relu')
        self.Q = tf.keras.layers.Dense(1, activation=None)

    def call(self, state, action):
        action = self.H1(tf.concat([state, action], axis=1))
        action = self.H2(action)
        Q = self.Q(action)
        return Q


class Actor(tf.keras.Model):
    """Defines a Actor Deep Learning Network"""

    def __init__(self, dim_actions, H1_dim=512, H2_dim=512, name='actor'):
        super(Actor, self).__init__()
        self.H1_dim = H1_dim
        self.H2_dim = H2_dim
        self.dim_actions = dim_actions
        self.model_name = name
        self.checkpoint = self.model_name + '.h5'
        self.H1 = tf.keras.layers.Dense(self.H1_dim, activation='relu')
        self.H2 = tf.keras.layers.Dense(self.H2_dim, activation='relu')
        self.mu = tf.keras.layers.Dense(self.dim_actions, activation='tanh')

    def call(self, state):
        action_prob = self.H1(state)
        action_prob = self.H2(action_prob)
        mu = self.mu(action_prob)
        return mu


class Agent:
    """Defines a RL Agent based on Actor-Critc method"""

    def __init__(self, env, name, alpha=0.001, beta=0.002,
                 gamma=0.99, tau=0.005,
                 H1=512, H2=256, batch_size=64, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = env.action_space.shape[0]
        self.obs_shape = env.observation_space.shape[0]
        self.batch_size = batch_size
        self.noise = noise
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.name = name

        self.actor = Actor(self.n_actions, name=self.name + 'actor')
        self.critic = Critic(self.n_actions, name=self.name + 'critic')
        self.target_actor = Actor(self.n_actions, name=self.name + 'target_actor')
        self.target_critic = Critic(self.n_actions, name=self.name + 'target_critic')

        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.update_networks(tau=1)

    def update_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    def save_models(self, path):
        self.actor.save_weights(path + self.actor.checkpoint)
        self.critic.save_weights(path + self.critic.checkpoint)
        self.target_actor.save_weights(path + self.target_actor.checkpoint)
        self.target_critic.save_weights(path + self.target_critic.checkpoint)

    def load_models(self, path):
        self.actor.load_weights(path + self.actor.checkpoint)
        self.critic.load_weights(path + self.critic.checkpoint)
        self.target_actor.load_weights(path + self.target_actor.checkpoint)
        self.target_critic.load_weights(path + self.target_critic.checkpoint)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        return actions[0]

    def learn(self, buffer, steps):
        for i in range(steps):
            if buffer.mem_cntr < self.batch_size:
                return

            state, action, reward, new_state, done = buffer.sample_buffer(self.batch_size)

            states = tf.convert_to_tensor(state, dtype=tf.float32)
            states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
            rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
            actions = tf.convert_to_tensor(action, dtype=tf.float32)

            with tf.GradientTape() as tape:
                target_actions = self.target_actor(states_)
                critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1)
                critic_value = tf.squeeze(self.critic(states, actions), 1)
                target = reward + self.gamma * critic_value_ * (1 - done)
                critic_loss = tf.keras.losses.MSE(target, critic_value)

            critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

            with tf.GradientTape() as tape:
                new_policy_actions = self.actor(states)
                actor_loss = -self.critic(states, new_policy_actions)
                actor_loss = tf.math.reduce_mean(actor_loss)

            actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))
            self.update_networks()
