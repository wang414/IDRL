from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
from gym.spaces.box import Box
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import core
from utils.logger import EpochLogger
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
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, idx, batch_size=32):
        idx = np.array([idx*3, idx*3+1, idx*3 + 2])
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs][:, idx],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


def method4(env_fn, env_name, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
          steps_per_epoch=4000, epochs=500, replay_size=int(1e6), gamma=0.99,
          polyak=0.995, pi_lr=1e-4, q_lr=1e-4, z_lr=5e-5, batch_size=100, start_steps=50000,
          update_after=2000, update_every=100, act_noise=0.1, num_test_episodes=10,
          max_ep_len=1000, logger_dir='logs', model_name='maqrdqn', save_freq=1, kappa=1.0, N=200,
          ucb=85, weight=0.5):
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
    if use_gpu:
        device = torch.device('cuda')

    logger = EpochLogger(logger_dir, model_name)
    logger.log_vars(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn() 
    
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
     # for special env ant
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    # ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    
    if env_name == 'HalfCheetah-v4' or 'Walker2d-v4':
        action_space_single = Box(low=-1, high=1, shape=[3,], dtype=np.float32)
        act_dim_sgl = action_space_single.shape[0]
        agent_num = 2

    ac = []
    for _ in range(agent_num):
        agent = actor_critic(env.observation_space, action_space_single, N, **ac_kwargs)
        if use_gpu:
            agent.cuda()
        ac.append(agent)
    ac_targ = deepcopy(ac)
    taus = torch.arange(
            0, N+1, device=device, dtype=torch.float32) / N
    tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, N)
    ac_targ = deepcopy(ac)
    logger.setup_pytorch_saver(ac)
    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for agent in ac_targ:
        for p in agent.parameters():
            p.requires_grad = False
    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac[0].pi, ac[0].q])
    print('\nNumber of parameters for each agent: \t pi: %d, \t q: %d\n'%var_counts)

    
    def calculate_huber_loss(td_errors, kappa=1.0):
        return torch.where(
            td_errors.abs() <= kappa,
            0.5 * td_errors.pow(2),
            kappa * (td_errors.abs() - 0.5 * kappa))


    def calculate_quantile_huber_loss(td_errors):
        assert not taus.requires_grad
        batch_size, N, N_dash = td_errors.shape

        # Calculate huber loss element-wisely.
        element_wise_huber_loss = calculate_huber_loss(td_errors, kappa)
        assert element_wise_huber_loss.shape == (
            batch_size, N, N_dash)

        # Calculate quantile huber loss element-wisely.
        element_wise_quantile_huber_loss = torch.abs(
            tau_hats[..., None] - (td_errors.detach() < 0).float()
            ) * element_wise_huber_loss / kappa
        assert element_wise_quantile_huber_loss.shape == (
            batch_size, N, N_dash)

        # Quantile huber loss.
        batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(
            dim=1).mean(dim=1, keepdim=True)
        assert batch_quantile_huber_loss.shape == (batch_size, 1)

        quantile_huber_loss = batch_quantile_huber_loss.mean()

        return quantile_huber_loss
    

    # Set up function for computing DDPG Q-loss
    def compute_loss_z(idx, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        if use_gpu:
            o = o.to(torch.device('cuda'))
            a = a.to(torch.device('cuda'))
            o2 = o2.to(torch.device('cuda'))
            r = r.to(torch.device('cuda'))
            d = d.to(torch.device('cuda'))

        # Calculate quantile values of current states and actions at taus.
        current_sa_quantiles = ac[idx].z(o, a)
        q = current_sa_quantiles.mean(dim=-1)
        current_sa_quantiles = current_sa_quantiles.unsqueeze(dim=-1)
        assert current_sa_quantiles.shape == (batch_size, N, 1)

        with torch.no_grad():
            # Calculate quantile values of next states and actions at tau_hats.
            next_sa_quantiles = ac_targ[idx].z(o2, ac_targ[idx].pi(o2)).unsqueeze(dim=-1).transpose(1, 2)
            assert next_sa_quantiles.shape == (batch_size, 1, N)

            # Calculate target quantile values.
            target_sa_quantiles = r.view(batch_size,1,1) + (
                1.0 - d.view(batch_size,1,1)) * gamma * next_sa_quantiles
            # print(target_sa_quantiles.shape)
            assert target_sa_quantiles.shape == (batch_size, 1, N)

        td_errors = target_sa_quantiles - current_sa_quantiles
        assert td_errors.shape == (batch_size, N, N)

        quantile_huber_loss = calculate_quantile_huber_loss(td_errors)

        if use_gpu:
            logs = q.cpu().detach().numpy().mean()
        else:
            logs = q.detach().numpy().mean()
        return quantile_huber_loss, logs

    def compute_loss_q(idx, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        if use_gpu:
            o = o.to(torch.device('cuda'))
            a = a.to(torch.device('cuda'))
            o2 = o2.to(torch.device('cuda'))
            r = r.to(torch.device('cuda'))
            d = d.to(torch.device('cuda'))

        q = ac[idx].q(o,a)


        # Bellman backup for Q function
        with torch.no_grad():

            q_pi_targ = ac_targ[idx].q(o2, ac_targ[idx].pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        if use_gpu:
            logs = q.cpu().detach().numpy().mean()
        else:
            logs = q.detach().numpy().mean()

        return loss_q, logs

    # Set up function for computing DDPG pi loss

    def compute_loss_pi(idx, data):
        o = data['obs']
        if use_gpu:
            o = o.to(torch.device('cuda'))
        z = ac[idx].z(o, ac[idx].pi(o))
        mean_z = z.mean(dim=1).reshape(-1, 1)
        abs_z = (z - mean_z).abs()
        min_idx = abs_z.argmin(dim=-1).long().reshape(-1, 1)
        #print('mean_z:\n', mean_z)
        #print('min_idx:\n', min_idx)
        #print('z[min_idx]:\n', z.gather(1, min_idx))
        
        q_pi = (1-weight) * ac[idx].q(o,ac[idx].pi(o)).squeeze(dim=-1) + \
               weight * ac[idx].z(o, ac[idx].pi(o))[:,ucb*2]
        return -q_pi.mean(), min_idx.cpu().numpy().mean(), z.detach().cpu().numpy().mean(axis=0)

    # Set up optimizers for policy and q-function
    pi_optimizers = []
    q_optimizers = []
    z_optimizers = []
    for i in range(agent_num):
        pi_optimizers.append(Adam(ac[i].pi.parameters(), lr=pi_lr))
        q_optimizers.append(Adam(ac[i].q.parameters(), lr=q_lr))
        z_optimizers.append(Adam(ac[i].z.parameters(), lr=z_lr))

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(idx, data):
        # First run one gradient descent step for Z.
        z_optimizers[idx].zero_grad()
        loss_z, loss_info_z = compute_loss_z(idx, data)
        loss_z.backward()
        z_optimizers[idx].step()

        # First run one gradient descent step for Q.
        q_optimizers[idx].zero_grad()
        loss_q, loss_info_q = compute_loss_q(idx, data)
        loss_q.backward()
        q_optimizers[idx].step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in ac[idx].q.parameters():
            p.requires_grad = False
        for p in ac[idx].z.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizers[idx].zero_grad()
        loss_pi, mean_idx, z = compute_loss_pi(idx, data)
        loss_pi.backward()
        pi_optimizers[idx].step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac[idx].q.parameters():
            p.requires_grad = True
        for p in ac[idx].z.parameters():
            p.requires_grad = True


        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac[idx].parameters(), ac_targ[idx].parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
        return loss_q.item(), loss_z.item(), loss_pi.item(), loss_info_q, loss_info_z, mean_idx, z

    def get_action(idx, o, noise_scale):
        o = torch.as_tensor(o, dtype=torch.float32)
        if use_gpu:
            o = o.to(torch.device('cuda'))
        a = ac[idx].act(obs=o, use_gpu=use_gpu)
        a += noise_scale * np.random.randn(act_dim_sgl)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        vals = np.ndarray([num_test_episodes], dtype=np.float64)
        lens = np.ndarray([num_test_episodes], dtype=np.int32)
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset()[0], False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                action = []
                for idx in range(agent_num):
                    action.append(get_action(idx, o, 0))
                action = np.concatenate(action, axis=-1)
                o, r, d, _, _ = test_env.step(action)
                ep_ret += r
                ep_len += 1
            vals[j] = ep_ret
            lens[j] = ep_len
        logger.store(test_avg_r=vals.mean(), test_std_r=vals.std())
        logger.store(test_avg_len=lens.mean())
        

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    # print(env.reset())
    o, ep_ret, ep_len, loss_q, loss_z, loss_pi, q_vals, z_vals, counts, mi, z= \
         env.reset(seed=seed)[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ep_rets = []
    test_env.reset(seed=np.random.randint(100))

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = []
            for idx in range(agent_num):
                a.append(get_action(idx, o, act_noise))
            a = np.concatenate(a, axis=-1)
        else:
            a = np.random.uniform(-act_limit, act_limit, act_dim)

        # Step the env
        o2, r, d, _, info = env.step(a)
        if DEBUG:
            print("o:\n{}\no2:\n{}\nr:\n{}\ndone:\n{}\ninfo:\n{}".format(o, o2, r, d, info))
        ep_len += 1
        ep_ret += r

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            ep_rets.append(ep_ret)
            o, ep_ret, ep_len = env.reset()[0], 0, 0
            

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                for idx in range(agent_num):
                    batch = replay_buffer.sample_batch(idx, batch_size)
                    lq, lz, lp, qv, zv, mean_idx ,z_= update(idx, data=batch)
                    loss_q += lq
                    loss_z += lz
                    loss_pi += lp
                    q_vals += qv
                    z_vals += zv
                    mi += mean_idx
                    z += z_
                    counts += 1

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            train_ret = np.array(ep_rets)
            ep_rets = []
            logger.store(Epoch=epoch)
            logger.store(train_avg_r=train_ret.mean(), train_std_r=train_ret.std())
            test_agent()
            # Test the performance of the deterministic version of the agent.
            logger.store(loss_Q = loss_q/counts, loss_Z = loss_z/counts, mean_idx = (int)(mi/counts), \
                loss_Pi = loss_pi/counts, Q_vals=q_vals/counts, Z_vals = z_vals/counts)
            loss_q, loss_z, loss_pi, q_vals, z_vals, counts, mi, z= \
                0, 0, 0, 0, 0, 0, 0, 0

            # Log info about epoch
            logger.logging()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v4')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ucb',type=int, default=85, help='upper quantile as the pi tartget, please make sure the value is bound in 100')
    parser.add_argument('--weight',type=float, default=0.5, help='weight of the element of ucb target')
    parser.add_argument('--exp_name', type=str, default='method4_v1')
    args = parser.parse_args()

    exp_name= args.env + args.exp_name + '_ucb{}_weight{}'.format(args.ucb, args.weight)

    logger_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')


    method4(lambda: gym.make(args.env), args.env, actor_critic=core.MLPActorCritic,
          ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
          gamma=args.gamma, seed=args.seed, epochs=args.epochs,
          logger_dir=logger_dir, model_name=exp_name, ucb=args.ucb, weight=args.weight)
