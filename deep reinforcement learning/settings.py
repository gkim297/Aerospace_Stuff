

class Settings:

    RUN_NAME = 'MichaelKim_default_run'
    ENVIRONMENT = 'kimg010914'
    RECORD_VIDEO = True
    VIDEO_RECORD_FREQUENCY = 20
    NOISELESS_AT_TEST_TIME = True
    LEARN_FROM_PIXELS = False
    RESUME_TRAINING = False
    USE_GPU_WHEN_AVAILABLE = True
    RANDOM_SEED = 13

    NUMBER_OF_ACTORS = 10
    NUMBER_OF_EPISODES = 5e4
    MAX_TRAINING_ITERATIONS = 1e6
    ACTOR_LEARNING_RATE = 0.0001
    CRITIC_LEARNING_RATE = 0.0001
    TARGET_NETWORK_TAU = 0.001
    NUMBER_OF_BINS = 51
    L2_REGULARIZATION = False
    L2_REG_PARAMETER = 1e-6

    UPDATE_TARGET_NETWORKS_EVERY_NUM_ITERATIONS = 1
    UPDATE_ACTORS_EVERY_NUM_EPISODES = 1
    CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES = 5
    LOG_TRAINING_PERFORMANCE_EVERY_NUM_ITERATIONS = 100
    DISPLAY_TRAINING_PERFORMANCE_EVERY_NUM_ITERATIONS = 50000
    DISPLAY_ACTOR_PERFORMANCE_EVERY_NUM_EPISODES = 2500

    PRIORITY_REPLAY_BUFFER = False
    PRIORITY_ALPHA = 0.6
    PRIORITY_BETA_START = 0.4
    PRIORITY_BETA_END = 1.0
    PRIORITY_EPSILON = 0.00001
    DUMP_PRIORITY_REPLAY_BUFFER_EVER_NUM_ITERATIONS = 200

    REPLAY_BUFFER_SIZE = 1000000
    REPLAY_BUFFER_START_TRAINING_FULLNESS = 0
    MINI_BATCH_SIZE = 256

    UNIFORM_OR_GAUSSIAN_NOISE = False
    if UNIFORM_OR_GAUSSIAN_NOISE:
        NOISE_SCALE = 1
    else:
        NOISE_SCALE = 1 / 3
    NOISE_SCALE_DECAY = 0.9999



    if LEARN_FROM_PIXELS:

        CONVOLUTIONAL_LAYERS = [{'filters': 32, 'kernel_size': [8, 8], 'strides': [4, 4]},
                                {'filters': 64, 'kernel_size': [4, 4], 'strides': [2, 2]},
                                {'filters': 64, 'kernel_size': [3, 3], 'strides': [1, 1]}]

    ACTOR_HIDDEN_LAYERS = [400, 300]
    CRITIC_HIDDEN_LAYERS = [400, 300]

    MODEL_SAVE_DIRECTORY = 'Tensorboard/Current/'
    TENSORBOARD_FILE_EXTENSION = '.tensorboard'
    SAVE_CHECKPOINT_EVERY_NUM_ITERATIONS = 100000

    environment_file = __import__('environment_' + ENVIRONMENT)
    if ENVIRONMENT == 'gym':
        env = environment_file.Environment('Temporary environment', 0, CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES,
                                           VIDEO_RECORD_FREQUENCY,
                                           MODEL_SAVE_DIRECTORY)
    else:
        env = environment_file.Environment()

    STATE_SIZE = env.STATE_SIZE
    UPPER_STATE_BOUND = env.UPPER_STATE_BOUND
    LOWER_STATE_BOUND = env.LOWER_STATE_BOUND
    ACTION_SIZE = env.ACTION_SIZE
    LOWER_ACTION_BOUND = env.LOWER_ACTION_BOUND
    UPPER_ACTION_BOUND = env.UPPER_ACTION_BOUND
    NORMALIZE_STATE = env.NORMALIZE_STATE
    MIN_V = env.MIN_V
    MAX_V = env.MAX_V
    DISCOUNT_FACTOR = env.DISCOUNT_FACTOR
    N_STEP_RETURN = env.N_STEP_RETURN
    TIMESTEP = env.TIMESTEP
    MAX_NUMBER_OF_TIMESTEPS = env.MAX_NUMBER_OF_TIMESTEPS
    IRRELEVANT_STATES = env.IRRELEVANT_STATES
    TEST_ON_DYNAMICS = env.TEST_ON_DYNAMICS
    KINEMATIC_NOISE = env.KINEMATIC_NOISE

    del env

    ACTION_RANGE = UPPER_ACTION_BOUND - LOWER_ACTION_BOUND  
    STATE_MEAN = (LOWER_STATE_BOUND + UPPER_STATE_BOUND) / 2.
    STATE_HALF_RANGE = (UPPER_STATE_BOUND - LOWER_STATE_BOUND) / 2.
