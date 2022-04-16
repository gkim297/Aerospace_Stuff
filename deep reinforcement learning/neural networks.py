import tensorflow as tf

from settings import Settings


class BuildActorNetwork:

    def __init__(self, state, scope):
        """
        The actor receives the state and outputs the action
        """

        self.__state = state
        self.__scope = scope

        with tf.variable_scope(self.__scope):

            self.__layer = self.__state

            if Settings.LEARN_FROM_PIXELS:

                for i, conv_layer_settings in enumerate(Settings.CONVOLUTIONAL_LAYERS):
                    self.__layer = tf.layers.conv2d(inputs=self.__layer,
                                                  activation=tf.nn.relu,
                                                  name='conv_layer' + str(i),
                                                  **conv_layer_settings)

                self.__layer = tf.layers.flatten(self.__layer)

            for i, number_of_neurons in enumerate(Settings.ACTOR_HIDDEN_LAYERS):
                self.__layer = tf.layers.dense(inputs=self.__layer,
                                             units=number_of_neurons,
                                             activation=tf.nn.relu,
                                             name='fully_connected_layer_' + str(i))


            self.__actions_out_unscaled = tf.layers.dense(inputs=self.__layer,
                                                        units=Settings.ACTION_SIZE,
                                                        activation=tf.nn.tanh,
                                                        name='output_layer')

            self.__action_scaled = tf.multiply(0.5, tf.multiply(self.__actions_out_unscaled,
                                                              Settings.ACTION_RANGE) + Settings.LOWER_ACTION_BOUND + Settings.UPPER_ACTION_BOUND)

            self.__parameters = tf.trainable_variables(scope=self.__scope)

    def generate_training_function(self, dQ_dAction):
        with tf.variable_scope(self.__scope):
            with tf.variable_scope('Training'):
                self.__optimizer = tf.train.AdamOptimizer(Settings.ACTOR_LEARNING_RATE)
                self.__actor_gradients = tf.gradients(self.__action_scaled, self.__parameters, -dQ_dAction)
                self.__actor_gradients_scaled = list(map(lambda x: tf.divide(x, Settings.MINI_BATCH_SIZE),
                                                       self.actor_gradients))
                actor_training_function = self.__optimizer.apply_gradients(
                    zip(self.__actor_gradients_scaled, self.__parameters))

                return actor_training_function


class BuildQNetwork:

    def __init__(self, state, action, scope):

        self.__state = state
        self.__action = action
        self.__scope = scope
        """
        Defines a critic network that predicts the q-distribution (expected return)
        from a given state and action. 

        The network archetectire is modified from the D4PG paper. The state goes through
        two layers on its own before being added to the action who has went through
        one layer. Then, the sum of the two goes through the final layer. Note: the 
        addition happend before the relu.
        """
        with tf.variable_scope(self.scope):
            self.__state_side = self.__state
            self.__action_side = self.__action

            if Settings.LEARN_FROM_PIXELS:
                for i, conv_layer_settings in enumerate(Settings.CONVOLUTIONAL_LAYERS):
                    self.__state_side = tf.layers.conv2d(inputs=self.__state_side,
                                                       activation=tf.nn.relu,
                                                       name='state_conv_layer' + str(i),
                                                       **conv_layer_settings)

                self.__state_side = tf.layers.flatten(self.__state_side)

            for i, number_of_neurons in enumerate(Settings.CRITIC_HIDDEN_LAYERS):
                self.__state_side = tf.layers.dense(inputs=self.__state_side,
                                                  units=number_of_neurons,
                                                  activation=None,
                                                  name='state_fully_connected_layer_' + str(i))
                if i < (len(Settings.CRITIC_HIDDEN_LAYERS) - 1):
                    self.__state_side = tf.nn.relu(self.__state_side)

            for i, number_of_neurons in enumerate(Settings.CRITIC_HIDDEN_LAYERS[1:]):
                self.__action_side = tf.layers.dense(inputs=self.__action_side,
                                                   units=number_of_neurons,
                                                   activation=None,
                                                   name='action_fully_connected_layer_' + str(i))
                if i < (len(Settings.CRITIC_HIDDEN_LAYERS) - 2):
                    self.__action_side = tf.nn.relu(self.__action_side)

            self.__layer = tf.add(self.__state_side, self.__action_side)
            self.__layer = tf.nn.relu(self.__layer)
            self.__q_distribution_logits = tf.layers.dense(inputs=self.__layer,
                                                         units=Settings.NUMBER_OF_BINS,
                                                         activation=None,
                                                         name='output_layer')

            self.__q_distribution = tf.nn.softmax(self.__q_distribution_logits, name='output_probabilities')
            self.__bins = tf.lin_space(Settings.MIN_V, Settings.MAX_V, Settings.NUMBER_OF_BINS)
            self.__parameters = tf.trainable_variables(scope=self.__scope)

            self.__dQ_dAction = tf.gradients(self.__q_distribution, self.__action,
                                           self.__bins)

    def generate_training_function(self, target_q_distribution, target_bins, importance_sampling_weights):
        with tf.variable_scope(self.__scope):
            with tf.variable_scope('Training'):
                self.__optimizer = tf.train.AdamOptimizer(Settings.CRITIC_LEARNING_RATE)
                projected_target_distribution = l2_project(target_bins, target_q_distribution, self.__bins)
                self.__loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.__q_distribution_logits,
                                                                       labels=tf.stop_gradient(
                                                                           projected_target_distribution))

                if Settings.PRIORITY_REPLAY_BUFFER:
                    self.__weighted_loss = self.__loss * importance_sampling_weights
                else:
                    self.__weighted_loss = self.__loss

                self.__mean_loss = tf.reduce_mean(self.__weighted_loss)

                if Settings.L2_REGULARIZATION:
                    self.__l2_reg_loss = tf.add_n(
                        [tf.nn.l2_loss(v) for v in self.__parameters if 'kernel' in v.name]) * Settings.L2_REG_PARAMETER
                else:
                    self.__l2_reg_loss = 0.0

                self.__total_loss = self.__mean_loss + self.__l2_reg_loss

                critic_training_function = self.__optimizer.minimize(self.__total_loss, var_list=self.__parameters)

                return critic_training_function, projected_target_distribution


def l2_project(z_p, p, z_q):
    """Projects distribution (z_p, p) onto support z_q under L2-metric over CDFs.
    The supports z_p and z_q are specified as tensors of distinct atoms (given
    in ascending order).
    Let Kq be len(z_q) and Kp be len(z_p). This projection works for any
    support z_q, in particular Kq need not be equal to Kp.
    Args:
      z_p: Tensor holding support of distribution p, shape `[batch_size, Kp]`.
      p: Tensor holding probability values p(z_p[i]), shape `[batch_size, Kp]`.
      z_q: Tensor holding support to project onto, shape `[Kq]`.
    Returns:
      Projection of (z_p, p) onto support z_q under Cramer distance.
    """

    vmin, vmax = z_q[0], z_q[-1]
    d_pos = tf.concat([z_q, vmin[None]], 0)[1:]
    d_neg = tf.concat([vmax[None], z_q], 0)[:-1]
    z_p = tf.clip_by_value(z_p, vmin, vmax)[:, None, :]

    d_pos = (d_pos - z_q)[None, :, None]
    d_neg = (z_q - d_neg)[None, :, None]
    z_q = z_q[None, :, None]

    d_neg = tf.where(d_neg > 0, 1. / d_neg, tf.zeros_like(d_neg))
    d_pos = tf.where(d_pos > 0, 1. / d_pos, tf.zeros_like(d_pos))

    delta_qp = z_p - z_q
    d_sign = tf.cast(delta_qp >= 0., dtype=p.dtype)

    delta_hat = (d_sign * delta_qp * d_pos) - ((1. - d_sign) * delta_qp * d_neg)
    p = p[:, None, :]
    return tf.reduce_sum(tf.clip_by_value(1. - delta_hat, 0., 1.) * p, 2)