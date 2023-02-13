from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
from gym.spaces.box import Box
import time
import core
from logx import EpochLogger
import warnings
warnings.filterwarnings('ignore')


DEBUG = False
use_gpu = torch.cuda.is_available()


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, idx, batch_size=32):
        idx = np.array([idx*2, idx*2+1])
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs][:, idx],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


def mac51(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=50, replay_size=int(1e6), gamma=0.99,
         polyak=0.995, pi_lr=3e-3, q_lr=3e-3, batch_size=100, start_steps=50000,
         update_after=2000, update_every=100, act_noise=0.1, num_test_episodes=10, 
         max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """
    Deep Deterministic Policy Gradient (DDPG)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of 
            actions as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_dim_sgl = 2 # for special env ant
    """
        for the task ant, we use a agent to control one leg
    """

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    # ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    action_space_single = Box(low=-1, high=1, shape=[2,], dtype=np.float32)
    ac = []
    for _ in range(4):
        agent = actor_critic(env.observation_space, action_space_single, **ac_kwargs)
        if use_gpu:
            agent.cuda()
        ac.append(agent)
    v_min = -5.0
    v_max = 15.0
    N = 51
    deltaZ = (v_max - v_min) / float(N - 1)
    z_range = np.array([v_min + i * deltaZ for i in range(N)])

    ac_targ = deepcopy(ac)
    z_range_torch = torch.from_numpy(z_range).type(torch.float32)
    if use_gpu:
        z_range_torch = z_range_torch.to(device=torch.device("cuda"))

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for agent in ac_targ:
        for p in agent.parameters():
            p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac[0].pi, ac[0].q])
    logger.log('\nNumber of parameters for each agent: \t pi: %d, \t q: %d\n'%var_counts)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(idx, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        if use_gpu:
            o = o.to(torch.device('cuda'))
            a = a.to(torch.device('cuda'))
            o2 = o2.to(torch.device('cuda'))
            r = r.to(torch.device('cuda'))
            d = d.to(torch.device('cuda'))

        data_num = o.shape[0]
        

        # action value distribution prediction
        z_eval = ac[idx].q(o,a)  # (m, N_ATOM)
        if use_gpu:
            q_eval = (z_eval.data.cpu().numpy() * z_range).sum(axis=-1)
        else:
            q_eval = (z_eval.data.numpy() * z_range).sum(axis=-1)
        # target distribution
        device = 'cuda' if use_gpu else 'cpu'
        z_target = torch.zeros((data_num, N), device=device)  # (m, N_ATOM)


        # Bellman backup for Z function
        with torch.no_grad():
            # q_pi_targ = ac_targ[idx].q(o2, ac_targ[idx].pi(o2))
            # backup = r + gamma * (1 - d) * q_pi_targ
            # get next state value
            z_next = ac_targ[idx].q(o2, ac_targ[idx].pi(o2))  # (m, N_ATOM)
            # categorical projection
            '''
            next_v_range : (z_j) i.e. values of possible return, shape : (m, N_ATOM)
            next_v_pos : relative position when offset of value is V_MIN, shape : (m, N_ATOM)
            '''
            # we vectorized the computation of support and position
            # print(z_range_torch.device)
            next_v_range = torch.unsqueeze(r, 1) + gamma * torch.unsqueeze((1. - d), 1) * torch.unsqueeze(z_range_torch, 0)
            next_v_pos = torch.zeros_like(next_v_range, device=device)
            # clip for categorical distribution
            next_v_range = torch.clip(next_v_range, v_min, v_max)
            # calc relative position of possible value
            next_v_pos = (next_v_range - v_min) / deltaZ
            # get lower/upper bound of relative position
            lb = torch.floor(next_v_pos).type(torch.int32)
            ub = torch.ceil(next_v_pos).type(torch.int32)
            # we didn't vectorize the computation of target assignment.
            for i in range(data_num):
                for j in range(N):
                    # calc prob mass of relative position weighted with distance
                    z_target[i, lb[i, j]] += (z_next * (ub - next_v_pos))[i, j]
                    z_target[i, ub[i, j]] += (z_next * (next_v_pos - lb))[i, j]

        # MSE loss against Bellman backup
        # print("z_target:\n{}\nz_eval:\n{}".format(z_target, z_eval))
        loss_q = - (z_target * torch.log(z_eval + 1e-6)).sum(dim=-1)  # (m)
        loss_q = torch.mean(loss_q)

        # Useful info for logging
        loss_info = dict(QVals=q_eval)

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(idx, data):
        o = data['obs']
        if use_gpu:
            o = o.to(torch.device('cuda'))
        q_pi = (ac[idx].q(o, ac[idx].pi(o)) * z_range_torch).sum(dim=-1)
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizers = []
    q_optimizers = []
    for i in range(4):
        pi_optimizers.append(Adam(ac[i].pi.parameters(), lr=pi_lr))
        q_optimizers.append(Adam(ac[i].q.parameters(), lr=q_lr))

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(idx, data):
        # First run one gradient descent step for Q.
        q_optimizers[idx].zero_grad()
        loss_q, loss_info = compute_loss_q(idx, data)
        loss_q.backward()
        q_optimizers[idx].step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in ac[idx].q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizers[idx].zero_grad()
        loss_pi = compute_loss_pi(idx, data)
        loss_pi.backward()
        pi_optimizers[idx].step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac[idx].q.parameters():
            p.requires_grad = True

        # Record things
        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac[idx].parameters(), ac_targ[idx].parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(idx, o, noise_scale):
        o = torch.as_tensor(o, dtype=torch.float32)
        if use_gpu:
            o = o.to(torch.device('cuda'))
        a = ac[idx].act(obs=o, use_gpu=use_gpu)
        a += noise_scale * np.random.randn(act_dim_sgl)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset()[0], False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                action = []
                for idx in range(4):
                    action.append(get_action(idx, o, 0))
                action = np.concatenate(action, axis=-1)
                o, r, d, _, _ = test_env.step(action)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    # print(env.reset())
    o, ep_ret, ep_len = env.reset()[0], 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = []
            for idx in range(4):
                a.append(get_action(idx, o, act_noise))
            a = np.concatenate(a, axis=-1)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _, info = env.step(a)
        if DEBUG:
            print("o:\n{}\no2:\n{}\nr:\n{}\ndone:\n{}\ninfo:\n{}".format(o, o2, r, d, info))
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset()[0], 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                for idx in range(4):
                    batch = replay_buffer.sample_batch(idx, batch_size)
                    update(idx, data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Ant-v4')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='mac51')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, './logs', True)

    mac51(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
