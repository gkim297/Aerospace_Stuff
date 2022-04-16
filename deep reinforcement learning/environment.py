import numpy as np
import os
import signal
import multiprocessing
from scipy.integrate import odeint  

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec


class Environment:

    def __init__(self):

        self.__TOTAL_STATE_SIZE = 8
        self.__IRRELEVANT_STATES = [0, 1, 2]
        self.__STATE_SIZE = self.__TOTAL_STATE_SIZE - len(self.__IRRELEVANT_STATES)
        self.__ACTION_SIZE = 3
        self.__LOWER_ACTION_BOUND = np.array([-0.1, -0.1,
                                            -10 * np.pi / 180])
        self.__UPPER_ACTION_BOUND = np.array([0.1, 0.1,
                                            10 * np.pi / 180])
        self.__LOWER_STATE_BOUND = np.array(
            [0., 0., -4 * 2 * np.pi, 0., 0., -4 * 2 * np.pi, 0., 0.])
        self.__UPPER_STATE_BOUND = np.array(
            [3.7, 2.4, 4 * 2 * np.pi, 3.7, 2.4, 4 * 2 * np.pi, 3.7, 2.4])
        self.__NORMALIZE_STATE = True
        self.__RANDOMIZE = True
        self.__NOMINAL_INITIAL_POSITION = np.array([3.0, 1.0, 0.0])
        self.__NOMINAL_TARGET_POSITION = np.array(
            [1.85, 1.2, 0])
        self.__MIN_V = -1000.
        self.__MAX_V = 100.
        self.__N_STEP_RETURN = 1
        self.__DISCOUNT_FACTOR = 0.95 ** (1 / self.__N_STEP_RETURN)
        self.__TIMESTEP = 0.2
        self.__TARGET_REWARD = 1.
        self.__FALL_OFF_TABLE_PENALTY = 0.
        self.__END_ON_FALL = False
        self.__GOAL_REWARD = 0.
        self.__NEGATIVE_PENALTY_FACTOR = 1.5
        self.__MAX_NUMBER_OF_TIMESTEPS = 900
        self.__ADDITIONAL_VALUE_INFO = False
        self.__REWARD_TYPE = True
        self.__REWARD_WEIGHTING = [0.5, 0.5, 0.1]
        self.__REWARD_MULTIPLIER = 250

        self.__USE_OBSTACLE = True
        self.__OBSTABLE_PENALTY = 15
        self.__OBSTABLE_DISTANCE = 0.2
        self.__OBSTACLE_INITIAL_POSITION = np.array([1.2, 1.2])
        self.__OBSTABLE_VELOCITY = np.array([0.0, 0.0])

        self.__TEST_ON_DYNAMICS = True
        self.__KINEMATIC_NOISE = False
        self.__KINEMATIC_NOISE_SD = [0.02, 0.02,
                                   np.pi / 100]
        self.__FORCE_NOISE_AT_TEST_TIME = False

        self.__KP = 0
        self.__KD = 2.0
        self.__CONTROLLER_ERROR_WEIGHT = [1, 1,
                                        0.05]

        self.__LENGTH = 0.3
        self.__MASS = 10
        self.__INERTIA = 1 / 12 * self.__MASS * (self.__LENGTH ** 2 + self.__LENGTH ** 2)

        self.__TARGET_COLLISION_DISTANCE = self.__LENGTH
        self.__TARGET_COLLISION_PENALTY = 15

        self.__PHASE_1_TIME = 90
        self.__DOCKING_TOO_FAST_PENALTY = 0
        self.__MAX_DOCKING_SPEED = [0.02, 0.02, 10]
        self.__TARGET_ANGULAR_VELOCITY = 0.0698
        self.__PENALIZE_VELOCITY = True
        self.__VELOCITY_PENALTY = [0.5, 0.5,
                                 0.0]


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, use_dynamics, test_time):
        """ NOTES:
               - if use_dynamics = True -> use dynamics
               - if test_time = True -> do not add "controller noise" to the kinematics
        """
        self.__dynamics_flag = False

        self.__phase_number = 0

        self.__test_time = test_time

        if self.__RANDOMIZE:
            self.__state = self.__NOMINAL_INITIAL_POSITION + np.random.randn(3) * [0.3, 0.3, np.pi / 2]
            self.__target_location = self.__NOMINAL_TARGET_POSITION + np.random.randn(3) * [0.15, 0.15, np.pi / 12]

        else:
            self.__state = self.__NOMINAL_INITIAL_POSITION
            self.__target_location = self.__NOMINAL_TARGET_POSITION

        self.__obstacle_location = self.__OBSTACLE_INITIAL_POSITION

        self.__docking_port = self.__target_location + np.array([np.cos(self.__target_location[2]) * (self.__LENGTH + 0.2),
                                                             np.sin(self.__target_location[2]) * (self.__LENGTH + 0.2),
                                                             -np.pi])

        self.__hold_point = self.__docking_port + np.array([np.cos(self.__target_location[2]) * (self.__LENGTH * 2 - 0.1),
                                                        np.sin(self.__target_location[2]) * (self.__LENGTH * 2 - 0.1), 0])

        self.__POSITION_STATE_LENGTH = len(self.__state)

        if use_dynamics:
            velocity_initial_conditions = np.array([0., 0., 0.])
            self.__state = np.concatenate((self.__state, velocity_initial_conditions))
            self.__dynamics_flag = True

        self.__time = 0.

        self.__previous_position_reward = [None, None, None]


    def step(self, action):

        if self.__dynamics_flag:
            kinematics_parameters = [action]

            guidance_propagation = odeint(kinematics_equations_of_motion, self.__state[:self.__POSITION_STATE_LENGTH],
                                          [self.__time, self.__time + self.__TIMESTEP], args=(kinematics_parameters,),
                                          full_output=0)

            guidance_position = guidance_propagation[1, :]

            control_effort = self.__controller(guidance_position,
                                             action)

            dynamics_parameters = [control_effort, self.__MASS, self.__INERTIA]

            next_states = odeint(dynamics_equations_of_motion, self.__state, [self.__time, self.__time + self.__TIMESTEP],
                                 args=(dynamics_parameters,), full_output=0)

            self.__state = next_states[1, :]

        else:

            kinematics_parameters = [action]

            guidance_position = []

            next_states = odeint(kinematics_equations_of_motion, self.__state, [self.__time, self.__time + self.__TIMESTEP],
                                 args=(kinematics_parameters,), full_output=0)

            self.__state = next_states[1, :]

            if self.__KINEMATIC___NOISE and (not self.__test_time or self.__FORCE_NOISE_AT_TEST_TIME):
                self.__state += np.random.randn(self.__POSITION_STATE_LENGTH) * self.__KINEMATIC_NOISE_SD


        self.__time += self.__TIMESTEP

        reward = self.__reward_function(action)

        done = self.__is_done()

        self.__check_phase_number()

        self.__obstacle_location += self.__OBSTABLE_VELOCITY * self.__TIMESTEP

        self.__target_location[2] += self.__TARGET_ANGULAR_VELOCITY * self.__TIMESTEP

        self.docking_port = self.__target_location + np.array([np.cos(self.__target_location[2]) * (self.__LENGTH + 0.2),
                                                             np.sin(self.__target_location[2]) * (self.__LENGTH + 0.2),
                                                             -np.pi])

        self.__hold_point = self.__docking_port + np.array([np.cos(self.__target_location[2]) * (self.__LENGTH * 2 - 0.1),
                                                        np.sin(self.__target_location[2]) * (self.__LENGTH * 2 - 0.1), 0])

        return self.__state, reward, done, guidance_position

    def check_phase_number(self):
        if self.__time >= self.__PHASE_1_TIME and self.__phase_number == 0:
            self.__phase_number = 1
            self.__previous_position_reward = [None, None, None]

    def controller(self, guidance_position, guidance_velocity):

        position_error = guidance_position - self.__state[:self.__POSITION_STATE_LENGTH]
        velocity_error = guidance_velocity - self.__state[self.__POSITION_STATE_LENGTH:]

        control_effort = self.__KP * position_error * self.__CONTROLLER_ERROR_WEIGHT + self.__KD * velocity_error * self.__CONTROLLER_ERROR_WEIGHT

        return control_effort

    def pose_error(self):
        """
        This method returns the pose error of the current state.
        Instead of returning [state, desired_state] as the state, I'll return
        [state, error]. The error will be more helpful to the policy I believe.
        """
        if self.__phase_number == 0:
            return self.__hold_point - self.__state[:self.__POSITION_STATE_LENGTH]
        elif self.__phase_number == 1:
            return self.docking_port - self.__state[:self.__POSITION_STATE_LENGTH]

    def reward_function(self, action):

        if self.__phase_number == 0:
            desired_location = self.__hold_point
        elif self.__phase_number == 1:
            desired_location = self.__docking_port

        current_position_reward = np.zeros(1)

        if self.__REWARD_TYPE:
            current_position_reward = -np.abs((desired_location - self.__state[
                                                                  :self.__POSITION_STATE_LENGTH]) * self.__REWARD_WEIGHTING) * self.__TARGET_REWARD
        else:
            current_position_reward = np.exp(-np.sum(np.absolute(desired_location - self.__state[
                                                                                    :self.__POSITION_STATE_LENGTH]) * self.__REWARD_WEIGHTING)) * self.__TARGET_REWARD

        reward = np.zeros(1)

        if np.all([self.__previous_position_reward[i] is not None for i in range(len(self.__previous_position_reward))]):
            reward = (current_position_reward - self.__previous_position_reward) * self.__REWARD_MULTIPLIER
            for i in range(len(reward)):
                if reward[i] < 0:
                    reward[i] *= self.__NEGATIVE_PENALTY_FACTOR

        self.__previous_position_reward = current_position_reward

        reward = np.sum(reward)

        if self.__phase_number == 1 and np.any(np.abs(action) > self.__MAX_DOCKING_SPEED):
            reward -= self.__DOCKING_TOO_FAST_PENALTY

        if self.state[0] > self.__UPPER_STATE_BOUND[0] or self.state[0] < self.LOWER_STATE_BOUND[0] or self.state[1] > \
                self.UPPER_STATE_BOUND[1] or self.state[1] < self.LOWER_STATE_BOUND[1]:
            reward -= self.FALL_OFF_TABLE_PENALTY / self.TIMESTEP

        if np.sum(np.absolute(self.state[:self.POSITION_STATE_LENGTH] - desired_location)) < 0.01:
            reward += self.GOAL_REWARD

        if np.linalg.norm(self.state[
                          :self.POSITION_STATE_LENGTH - 1] - self.obstacle_location) <= self.OBSTABLE_DISTANCE and self.USE_OBSTACLE:
            reward -= self.OBSTABLE_PENALTY

        if np.linalg.norm(self.state[:self.POSITION_STATE_LENGTH - 1] - self.target_location[
                                                                        :-1]) <= self.TARGET_COLLISION_DISTANCE:
            reward -= self.TARGET_COLLISION_PENALTY

        if self.PENALIZE_VELOCITY:
            radius = np.linalg.norm(
                desired_location[:2] - self.target_location[:2])
            reference_velocity = self.TARGET_ANGULAR_VELOCITY * np.array(
                [-radius * np.sin(self.target_location[2]), radius * np.cos(self.target_location[2]), 1])
            reward -= np.sum(
                np.abs(action - reference_velocity) / (self.pose_error() ** 2 + 0.01) * self.VELOCITY_PENALTY)

        return (reward * self.TIMESTEP).squeeze()

    def is_done(self):
        """
            NOTE: THE ENVIRONMENT MUST RETURN done = True IF THE EPISODE HAS
                  REACHED ITS LAST TIMESTEP
        """

        if self.state[0] > self.UPPER_STATE_BOUND[0] or self.state[0] < self.LOWER_STATE_BOUND[0] or self.state[1] > \
                self.UPPER_STATE_BOUND[1] or self.state[1] < self.LOWER_STATE_BOUND[1]:
            done = self.END_ON_FALL
        else:
            done = False

        if self.state[2] > self.UPPER_STATE_BOUND[2] or self.state[2] < self.LOWER_STATE_BOUND[2]:
            pass

        if round(self.time / self.TIMESTEP) == self.MAX_NUMBER_OF_TIMESTEPS:
            done = True

        return done

    def generate_queue(self):
        self.agent_to_env = multiprocessing.Queue(maxsize=1)
        self.env_to_agent = multiprocessing.Queue(maxsize=1)

        return self.agent_to_env, self.env_to_agent

    def obstable_relative_location(self):
        relative_position = self.obstacle_location - self.state[:self.POSITION_STATE_LENGTH - 1]

        return relative_position

    def run(self):
        """
        This method is called when the environment process is launched by main.py.
        It is responsible for continually listening for an input action from the
        agent through a Queue. If an action is received, it is to step the environment
        and return the results.
        """

        signal.signal(signal.SIGINT, signal.SIG_IGN)

        while True:
            action, *test_time = self.agent_to_env.get()

            if type(action) == bool:
                self.reset(action, test_time[0])
                self.env_to_agent.put((np.append(self.state[:self.POSITION_STATE_LENGTH],
                                                 np.append(self.pose_error(), self.obstable_relative_location())),
                                       self.target_location))

            else:

                next_state, reward, done, *guidance_position = self.step(action)

                # Return the results
                self.env_to_agent.put((np.append(next_state[:self.POSITION_STATE_LENGTH],
                                                 np.append(self.pose_error(), self.obstable_relative_location())),
                                       reward, done, guidance_position))


def kinematics_equations_of_motion(state, t, parameters):

    action = parameters[0]

    derivatives = action

    return derivatives


def dynamics_equations_of_motion(state, t, parameters):

    x, y, theta, xdot, ydot, thetadot = state
    control_effort, mass, inertia = parameters

    derivatives = np.array((xdot, ydot, thetadot, control_effort[0] / mass, control_effort[1] / mass,
                            control_effort[2] / inertia)).squeeze()

    return derivatives


def render(states, actions, desired_pose, instantaneous_reward_log, cumulative_reward_log, critic_distributions,
           target_critic_distributions, projected_target_distribution, bins, loss_log, guidance_position_log,
           episode_number, filename, save_directory):
    temp_env = Environment()

    extra_information = temp_env.ADDITIONAL_VALUE_INFO

    x, y, theta = states[:, 0], states[:, 1], states[:, 2]

    length = temp_env.LENGTH

    r1_b = length / 2. * np.array([[1.], [1.]])  # [2, 1]
    r2_b = length / 2. * np.array([[1.], [-1.]])
    r3_b = length / 2. * np.array([[-1.], [-1.]])
    r4_b = length / 2. * np.array([[-1.], [1.]])

    C_Ib = np.moveaxis(np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]]), source=2, destination=0)

    r1_I = np.matmul(C_Ib, r1_b)
    r2_I = np.matmul(C_Ib, r2_b)
    r3_I = np.matmul(C_Ib, r3_b)
    r4_I = np.matmul(C_Ib, r4_b)


    target_angles = desired_pose[2] + [temp_env.TARGET_ANGULAR_VELOCITY * i * temp_env.TIMESTEP for i in
                                       range(len(theta))]

    C_Ib = np.moveaxis(np.array([[np.cos(target_angles), -np.sin(target_angles)],
                                 [np.sin(target_angles), np.cos(target_angles)]]), source=2,
                       destination=0)

    r1_des = np.matmul(C_Ib, r1_b)  # [2, 1]
    r2_des = np.matmul(C_Ib, r2_b)  # [2, 1]
    r3_des = np.matmul(C_Ib, r3_b)  # [2, 1]
    r4_des = np.matmul(C_Ib, r4_b)  # [2, 1]

    obstacle_x = temp_env.OBSTACLE_INITIAL_POSITION[0] + [temp_env.OBSTABLE_VELOCITY[0] * i * temp_env.TIMESTEP for i in
                                                          range(len(theta))]
    obstacle_y = temp_env.OBSTACLE_INITIAL_POSITION[1] + [temp_env.OBSTABLE_VELOCITY[1] * i * temp_env.TIMESTEP for i in
                                                          range(len(theta))]


    figure = plt.figure(constrained_layout=True)
    figure.set_size_inches(5, 4, True)

    if extra_information:
        grid_spec = gridspec.GridSpec(nrows=2, ncols=3, figure=figure)
        subfig1 = figure.add_subplot(grid_spec[0, 0], aspect='equal', autoscale_on=False, xlim=(0, 3.4), ylim=(0, 2.4))
        subfig2 = figure.add_subplot(grid_spec[0, 1], xlim=(np.min([np.min(instantaneous_reward_log), 0]) - (
                    np.max(instantaneous_reward_log) - np.min(instantaneous_reward_log)) * 0.02,
                                                            np.max([np.max(instantaneous_reward_log), 0]) + (
                                                                        np.max(instantaneous_reward_log) - np.min(
                                                                    instantaneous_reward_log)) * 0.02),
                                     ylim=(-0.5, 0.5))
        subfig3 = figure.add_subplot(grid_spec[0, 2], xlim=(np.min(loss_log) - 0.01, np.max(loss_log) + 0.01),
                                     ylim=(-0.5, 0.5))
        subfig4 = figure.add_subplot(grid_spec[1, 0], ylim=(0, 1.02))
        subfig5 = figure.add_subplot(grid_spec[1, 1], ylim=(0, 1.02))
        subfig6 = figure.add_subplot(grid_spec[1, 2], ylim=(0, 1.02))

        subfig1.set_xlabel("X Position (m)", fontdict={'fontsize': 8})
        subfig1.set_ylabel("Y Position (m)", fontdict={'fontsize': 8})
        subfig2.set_title("Timestep Reward", fontdict={'fontsize': 8})
        subfig3.set_title("Current loss", fontdict={'fontsize': 8})
        subfig4.set_title("Q-dist", fontdict={'fontsize': 8})
        subfig5.set_title("Target Q-dist", fontdict={'fontsize': 8})
        subfig6.set_title("Bellman projection", fontdict={'fontsize': 8})

        subfig1.tick_params(labelsize=8)
        subfig2.tick_params(which='both', left=False, labelleft=False, labelsize=8)
        subfig3.tick_params(which='both', left=False, labelleft=False, labelsize=8)
        subfig4.tick_params(which='both', left=False, labelleft=False, right=True, labelright=False, labelsize=8)
        subfig5.tick_params(which='both', left=False, labelleft=False, right=True, labelright=False, labelsize=8)
        subfig6.tick_params(which='both', left=False, labelleft=False, right=True, labelright=True, labelsize=8)

        subfig4.grid(True)
        subfig5.grid(True)
        subfig6.grid(True)

        subfig2.set_xticks([np.min(instantaneous_reward_log), 0, np.max(instantaneous_reward_log)] if np.sign(
            np.min(instantaneous_reward_log)) != np.sign(np.max(instantaneous_reward_log)) else [
            np.min(instantaneous_reward_log), np.max(instantaneous_reward_log)])
        subfig3.set_xticks([np.min(loss_log), np.max(loss_log)])
        subfig4.set_xticks([bins[i * 5] for i in range(round(len(bins) / 5) + 1)])
        subfig4.tick_params(axis='x', labelrotation=-90)
        subfig4.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
        subfig5.set_xticks([bins[i * 5] for i in range(round(len(bins) / 5) + 1)])
        subfig5.tick_params(axis='x', labelrotation=-90)
        subfig5.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
        subfig6.set_xticks([bins[i * 5] for i in range(round(len(bins) / 5) + 1)])
        subfig6.tick_params(axis='x', labelrotation=-90)
        subfig6.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])

    else:
        subfig1 = figure.add_subplot(1, 1, 1, aspect='equal', autoscale_on=False, xlim=(0, 3.4), ylim=(0, 2.4),
                                     xlabel='X Position (m)', ylabel='Y Position (m)')

    body, = subfig1.plot([], [], color='k', linestyle='-', linewidth=2)
    front_face, = subfig1.plot([], [], color='g', linestyle='-', linewidth=2)
    body_dot = subfig1.scatter([], [], color='r')

    if extra_information:
        reward_bar = subfig2.barh(y=0, height=0.2, width=0)
        loss_bar = subfig3.barh(y=0, height=0.2, width=0)
        q_dist_bar = subfig4.bar(x=bins, height=np.zeros(shape=len(bins)), width=bins[1] - bins[0])
        target_q_dist_bar = subfig5.bar(x=bins, height=np.zeros(shape=len(bins)), width=bins[1] - bins[0])
        projected_q_dist_bar = subfig6.bar(x=bins, height=np.zeros(shape=len(bins)), width=bins[1] - bins[0])
        time_text = subfig1.text(x=0.2, y=0.91, s='', fontsize=8, transform=subfig1.transAxes)
        reward_text = subfig1.text(x=0.0, y=1.02, s='', fontsize=8, transform=subfig1.transAxes)
    else:
        time_text = subfig1.text(x=0.03, y=0.96, s='', fontsize=8, transform=subfig1.transAxes)
        reward_text = subfig1.text(x=0.62, y=0.96, s='', fontsize=8, transform=subfig1.transAxes)
        episode_text = subfig1.text(x=0.40, y=1.02, s='', fontsize=8, transform=subfig1.transAxes)

    desired_pos, = subfig1.plot([], [], color='r', linestyle='-', linewidth=2)
    desired_pos_front, = subfig1.plot([], [], color='k', linestyle='-', linewidth=2)

    def initialize_axes():
        body.set_data([], [])
        front_face.set_data([], [])
        time_text.set_text('')

        if not extra_information:
            episode_text.set_text('Episode ' + str(episode_number))

        return body, front_face, time_text, body_dot

    def render_one_frame(frame, *fargs):
        temp_env = fargs[0]

        thisx = [r2_I[frame, 0, 0], r3_I[frame, 0, 0], r4_I[frame, 0, 0], r1_I[frame, 0, 0]] + x[frame]
        thisy = [r2_I[frame, 1, 0], r3_I[frame, 1, 0], r4_I[frame, 1, 0], r1_I[frame, 1, 0]] + y[frame]
        body.set_data(thisx, thisy)

        thisx = [r1_I[frame, 0, 0], r2_I[frame, 0, 0]] + x[frame]
        thisy = [r1_I[frame, 1, 0], r2_I[frame, 1, 0]] + y[frame]
        front_face.set_data(thisx, thisy)

        thisx = [r2_des[frame, 0, 0], r3_des[frame, 0, 0], r4_des[frame, 0, 0], r1_des[frame, 0, 0]] + desired_pose[0]
        thisy = [r2_des[frame, 1, 0], r3_des[frame, 1, 0], r4_des[frame, 1, 0], r1_des[frame, 1, 0]] + desired_pose[1]
        desired_pos.set_data(thisx, thisy)

        thisx = [r1_des[frame, 0, 0], r2_des[frame, 0, 0]] + desired_pose[0]
        thisy = [r1_des[frame, 1, 0], r2_des[frame, 1, 0]] + desired_pose[1]
        desired_pos_front.set_data(thisx, thisy)

        body_dot.set_offsets(np.hstack((x[frame], y[frame])))
        if frame != 0:
            subfig1.patches.clear()
        time_text.set_text('Time = %.1f s' % (frame * temp_env.TIMESTEP))

        if temp_env.USE_OBSTACLE:
            obstacle_dot = plt.Circle((obstacle_x[frame], obstacle_y[frame]), radius=np.max(
                [0.01, temp_env.OBSTABLE_DISTANCE - np.sqrt(2) * temp_env.LENGTH / 2]), fill=False, color='k')
            subfig1.add_patch(obstacle_dot)

        reward_text.set_text('Total reward = %.1f' % cumulative_reward_log[frame])

        if extra_information:
            reward_bar[0].set_width(instantaneous_reward_log[frame])
            if instantaneous_reward_log[frame] < 0:
                reward_bar[0].set_color('r')
            else:
                reward_bar[0].set_color('g')

            loss_bar[0].set_width(loss_log[frame])

            for this_bar, new_value in zip(q_dist_bar, critic_distributions[frame, :]):
                this_bar.set_height(new_value)

            for this_bar, new_value in zip(target_q_dist_bar, target_critic_distributions[frame, :]):
                this_bar.set_height(new_value)

            for this_bar, new_value in zip(projected_q_dist_bar, projected_target_distribution[frame, :]):
                this_bar.set_height(new_value)

        if temp_env.TEST_ON_DYNAMICS:
            position_arrow = plt.Arrow(x[frame], y[frame], guidance_position_log[frame, 0] - x[frame],
                                       guidance_position_log[frame, 1] - y[frame], width=0.06, color='k')

            subfig1.add_patch(position_arrow)

        return body, front_face, time_text, body_dot

    fargs = [temp_env]
    animator = animation.FuncAnimation(figure, render_one_frame,
                                       frames=np.linspace(0, len(states) - 1, len(states)).astype(int),
                                       blit=True, init_func=initialize_axes, fargs=fargs)

    try:
        animator.save(filename=filename + '_episode_' + str(episode_number) + '.mp4', fps=30, dpi=100)
        os.makedirs(os.path.dirname(save_directory + filename + '/videos/'), exist_ok=True)
        os.rename(filename + '_episode_' + str(episode_number) + '.mp4',
                  save_directory + filename + '/videos/episode_' + str(episode_number) + '.mp4')
    except:
        print("Skipping animation for episode %i due to an error" % episode_number)
        try:
            os.remove(filename + '_episode_' + str(episode_number) + '.mp4')
        except:
            pass

    del temp_env
    plt.close(figure)