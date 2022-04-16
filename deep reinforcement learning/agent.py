
import time
import tensorflow as tf
import numpy as np
import os
import queue
from collections import deque
from pyvirtualdisplay import Display

from settings import Settings
from neural networks import BuildActorNetwork
environment_file = __import__('environment_' + Settings.ENVIRONMENT)


class Agent:

    def __init__(self, sess, n_agent, agent_to_env, env_to_agent, replay_buffer, writer, filename, learner_policy_parameters, agent_to_learner, learner_to_agent):

        print("Initializing agent " + str(n_agent) + "...")

        self.__n_agent = n_agent
        self.__sess = sess
        self.__replay_buffer = replay_buffer
        self.__filename = filename
        self.__learner_policy_parameters = learner_policy_parameters
        self.__agent_to_env = agent_to_env
        self.__env_to_agent = env_to_agent
        self.__agent_to_learner = agent_to_learner
        self.__learner_to_agent = learner_to_agent

        self.__build_actor()

        self.__build_actor_update_operation()

        self.__create_summary_functions()
        self.__writer = writer

        if Settings.RECORD_VIDEO and self.__n_agent == 1:
            self.__display = Display(visible = False, size = (1400,900))
            self.__display.start()

        print("Agent %i initialized!" % self.__n_agent)


    def create_summary_functions(self):
        self.__timestep_number_placeholder      = tf.placeholder(tf.float32)
        self.__episode_reward_placeholder       = tf.placeholder(tf.float32)
        timestep_number_summary               = tf.summary.scalar("Agent_" + str(self.__n_agent) + "/Number_of_timesteps", self.__timestep_number_placeholder)
        episode_reward_summary                = tf.summary.scalar("Agent_" + str(self.vn_agent) + "/Episode_reward", self.__episode_reward_placeholder)
        self.__regular_episode_summary          = tf.summary.merge([timestep_number_summary, episode_reward_summary])

        if self.__n_agent == 1:
            test_time_episode_reward_summary  = tf.summary.scalar("Test_agent/Episode_reward", self.__episode_reward_placeholder)
            test_time_timestep_number_summary = tf.summary.scalar("Test_agent/Number_of_timesteps", self.__timestep_number_placeholder)
            self.__test_time_episode_summary    = tf.summary.merge([test_time_episode_reward_summary, test_time_timestep_number_summary])


    def build_actor(self):
        agent_name = 'agent_' + str(self.__n_agent)
        self.__state_placeholder = tf.placeholder(dtype = tf.float32, shape = [None, Settings.STATE_SIZE], name = 'state_placeholder')

        self.__policy = BuildActorNetwork(self.__state_placeholder, scope = agent_name)


    def build_actor_update_operation(self):
        update_operations = []
        source_variables = self.__learner_policy_parameters
        destination_variables = self.__policy.parameters

        for source_variable, destination_variable in zip(source_variables, destination_variables):
            update_operations.append(destination_variable.assign(source_variable))

        self.__update_actor_parameters = update_operations

    def run(self, stop_run_flag, replay_buffer_dump_flag, starting_episode_number):
        print("Starting to run agent %i at episode %i." % (self.__n_agent, starting_episode_number[self.__n_agent -1]))

        self.__sess.run(self.__update_actor_parameters)

        episode_number = starting_episode_number[self.__n_agent - 1]

        noise_scale = 0.

        start_time = time.time()

        self.__n_step_memory = deque()

        while episode_number <= Settings.NUMBER_OF_EPISODES and not stop_run_flag.is_set():

            self.__n_step_memory.clear()


            test_time = (self.__n_agent == 1) and (episode_number % Settings.CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES == 0 or episode_number == 1)

            if test_time and Settings.TEST_ON_DYNAMICS:
                self.__agent_to_env.put((True, test_time))
            else:
                self.__agent_to_env.put((False, test_time))
            state, desired_pose = self.__env_to_agent.get()


            if test_time:
                if Settings.NOISELESS_AT_TEST_TIME:
                    noise_scale = 0

                if Settings.RECORD_VIDEO and (episode_number % (Settings.CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES*Settings.VIDEO_RECORD_FREQUENCY) == 0 or episode_number == 1):
                    raw_state_log = []
                    state_log = []
                    action_log = []
                    next_state_log = []
                    instantaneous_reward_log = []
                    cumulative_reward_log = []
                    done_log = []
                    discount_factor_log = []
                    guidance_position_log = []
                    raw_state_log.append(state)

            else:

                noise_scale = Settings.NOISE_SCALE * Settings.NOISE_SCALE_DECAY ** episode_number


            if Settings.NORMALIZE_STATE:
                state = (state - Settings.STATE_MEAN)/Settings.STATE_HALF_RANGE

            state = np.delete(state, Settings.IRRELEVANT_STATES)

            episode_reward = 0
            timestep_number = 0
            done = False

            while not done:

                action = self.__sess.run(self.__policy.action_scaled, feed_dict = {self.__state_placeholder: np.expand_dims(state,0)})[0]

                if Settings.UNIFORM_OR_GAUSSIAN_NOISE:
                    exploration_noise = np.random.uniform(low = -Settings.ACTION_RANGE, high = Settings.ACTION_RANGE, size = Settings.ACTION_SIZE)*noise_scale
                else:
                    exploration_noise = np.random.normal(size = Settings.ACTION_SIZE)*Settings.ACTION_RANGE*noise_scale

                action = np.clip(action + exploration_noise, Settings.LOWER_ACTION_BOUND, Settings.UPPER_ACTION_BOUND)


                self.__agent_to_env.put((action,))

                next_state, reward, done, *guidance_position = self.__env_to_agent.get()

                episode_reward += reward

                if self.__n_agent == 1 and Settings.RECORD_VIDEO and (episode_number % (Settings.CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES*Settings.VIDEO_RECORD_FREQUENCY) == 0 or episode_number == 1) and not Settings.ENVIRONMENT == 'gym':
                    if not done:
                        raw_state_log.append(next_state)

                if Settings.NORMALIZE_STATE:
                    next_state = (next_state - Settings.STATE_MEAN)/Settings.STATE_HALF_RANGE

                next_state = np.delete(next_state, Settings.IRRELEVANT_STATES)

                self.__n_step_memory.append((state, action, reward))

                if (len(self.__n_step_memory) >= Settings.N_STEP_RETURN):
                    state_0, action_0, reward_0 = self.__n_step_memory.popleft()
                    n_step_reward = reward_0
                    discount_factor = Settings.DISCOUNT_FACTOR
                    for (state_i, action_i, reward_i) in self.__n_step_memory:
                        n_step_reward += reward_i*discount_factor
                        discount_factor *= Settings.DISCOUNT_FACTOR


                    self.__replay_buffer.add((state_0, action_0, n_step_reward, next_state, done, discount_factor))

                    if self.__n_agent == 1 and Settings.RECORD_VIDEO and (episode_number % (Settings.CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES*Settings.VIDEO_RECORD_FREQUENCY) == 0 or episode_number == 1) and not Settings.ENVIRONMENT == 'gym':
                        state_log.append(state_0)
                        action_log.append(action_0)
                        next_state_log.append(next_state)
                        cumulative_reward_log.append(episode_reward)
                        instantaneous_reward_log.append(n_step_reward)
                        done_log.append(done)
                        discount_factor_log.append(discount_factor)
                        guidance_position_log.append(guidance_position)

                state = next_state
                timestep_number += 1


                if done:
                    while len(self.__n_step_memory) > 0:
                        state_0, action_0, reward_0 = self.__n_step_memory.popleft()
                        n_step_reward = reward_0
                        discount_factor = Settings.DISCOUNT_FACTOR
                        for (state_i, action_i, reward_i) in self.__n_step_memory:
                            n_step_reward += reward_i*discount_factor
                            discount_factor *= Settings.DISCOUNT_FACTOR

                        replay_buffer_dump_flag.wait()
                        self.__replay_buffer.add((state_0, action_0, n_step_reward, next_state, done, discount_factor))

                        if self.__n_agent == 1 and Settings.RECORD_VIDEO and (episode_number % (Settings.CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES*Settings.VIDEO_RECORD_FREQUENCY) == 0 or episode_number == 1) and not Settings.ENVIRONMENT == 'gym':
                            state_log.append(state_0)
                            action_log.append(action_0)
                            next_state_log.append(next_state)
                            cumulative_reward_log.append(episode_reward)
                            instantaneous_reward_log.append(n_step_reward)
                            done_log.append(done)
                            discount_factor_log.append(discount_factor)
                            guidance_position_log.append(guidance_position)


            if self.__n_agent == 1 and Settings.RECORD_VIDEO and (episode_number % (Settings.CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES*Settings.VIDEO_RECORD_FREQUENCY) == 0 or episode_number == 1) and not Settings.ENVIRONMENT == 'gym':
                print("Rendering Actor %i at episode %i" % (self.__n_agent, episode_number))

                os.makedirs(os.path.dirname(Settings.MODEL_SAVE_DIRECTORY + self.__filename + '/trajectories/'), exist_ok=True)
                np.savetxt(Settings.MODEL_SAVE_DIRECTORY + self.__filename + '/trajectories/' + str(episode_number) + '.txt',np.asarray(raw_state_log))

                self.__agent_to_learner.put((np.asarray(state_log), np.asarray(action_log), np.asarray(next_state_log), np.asarray(instantaneous_reward_log), np.asarray(done_log), np.asarray(discount_factor_log)))

                try:
                    critic_distributions, target_critic_distributions, projected_target_distribution, loss_log = self.__learner_to_agent.get(timeout = 3)

                    bins = np.linspace(Settings.MIN_V, Settings.MAX_V, Settings.NUMBER_OF_BINS)

                    environment_file.render(np.asarray(raw_state_log), np.asarray(action_log), desired_pose, np.asarray(instantaneous_reward_log), np.asarray(cumulative_reward_log), critic_distributions, target_critic_distributions, projected_target_distribution, bins, np.asarray(loss_log), np.squeeze(np.asarray(guidance_position_log)), episode_number, self.filename, Settings.MODEL_SAVE_DIRECTORY)

                except queue.Empty:
                    print("Skipping this animation!")
                    raise SystemExit

            if episode_number % Settings.UPDATE_ACTORS_EVERY_NUM_EPISODES == 0:
                self.__sess.run(self.__update_actor_parameters)

            if episode_number % Settings.DISPLAY_ACTOR_PERFORMANCE_EVERY_NUM_EPISODES == 0:
                print("Actor " + str(self.__n_agent) + " ran " + str(Settings.DISPLAY_ACTOR_PERFORMANCE_EVERY_NUM_EPISODES) + " episodes in %.1f minutes, and is now at episode %i" % ((time.time() - start_time)/60, episode_number))
                start_time = time.time()


            feed_dict = {self.__episode_reward_placeholder:  episode_reward, self.__timestep_number_placeholder: timestep_number}
            if test_time:
                summary = self.__sess.run(self.__test_time_episode_summary, feed_dict = feed_dict)
            else:
                summary = self.__sess.run(self.__regular_episode_summary,   feed_dict = feed_dict)
            self.__writer.add_summary(summary, episode_number)

            episode_number += 1

        if Settings.RECORD_VIDEO and self.__n_agent == 1:
            self.__display.stop()

        print("Actor %i finished after running %i episodes!" % (self.__n_agent, episode_number - 1))