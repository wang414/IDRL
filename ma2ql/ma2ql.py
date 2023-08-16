from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
from gym.spaces.box import Box
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.logger import EpochLogger
from utils import core
import warnings
warnings.filterwarnings('ignore')

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, single_act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.act_dim_sgl = single_act_dim

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, idx, batch_size=32):
        idx = np.array([idx * self.act_dim_sgl + i for i in range(self.act_dim_sgl)])
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs][:, idx],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


use_gpu = torch.cuda.is_available()

def ddpg(args, env_fn, env_name, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
         polyak=0.995, pi_lr=1e-4, q_lr=1e-4, batch_size=100, start_steps=10000, 
         update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, 
         max_ep_len=1000, logger_dir='logs', model_name='ma2ql'):
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
    
    #batch_size = batch_size * 5
    device = torch.device("cuda")
    torch.set_num_threads(1)
    
    logger = EpochLogger(logger_dir, model_name)
    logger.log_vars(locals())

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]
    if env_name == 'Hopper-v4':
        action_space_single = Box(low=-act_limit, high=act_limit, shape=[1,], dtype=np.float32)
        act_dim_sgl = action_space_single.shape[0]
        agent_num = 3
    elif env_name == 'HalfCheetah-v4':
        action_space_single = Box(low=-act_limit, high=act_limit, shape=[3,], dtype=np.float32)
        act_dim_sgl = action_space_single.shape[0]
        agent_num = 2
    elif env_name == 'Walker2d-v4':
        action_space_single = Box(low=-act_limit, high=act_limit, shape=[2,], dtype=np.float32)
        act_dim_sgl = action_space_single.shape[0]
        agent_num = 3
    elif env_name == 'Ant-v4':
        action_space_single = Box(low=-act_limit, high=act_limit, shape=[2,], dtype=np.float32)
        act_dim_sgl = action_space_single.shape[0]
        agent_num = 4
    elif env_name == 'Humanoid-v4':
        action_space_single = Box(low=-act_limit, high=act_limit, shape=[1,], dtype=np.float32)
        act_dim_sgl = action_space_single.shape[0]
        agent_num = 17

    # Create actor-critic module and target networks
    ac = []
    for _ in range(agent_num):
        agent = actor_critic(env.observation_space, action_space_single, **ac_kwargs)
        if use_gpu:
            agent.cuda()
        ac.append(agent)
    
    ac_targ = deepcopy(ac)
    logger.setup_pytorch_saver(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for targ in ac_targ:
        for p in targ.parameters():
            p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim, act_dim_sgl, replay_size)
        
    # AgentBy
    if args.update_agent > 0:
        update_policy_id = 0
    else:
        update_policy_id = -1

    # Count variables (protip: try to get a feel for how different size networks behave!)
    #var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    #logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)
    

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data, p_id=0):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        if use_gpu:
            o = o.to(torch.device('cuda'))
            a = a.to(torch.device('cuda'))
            o2 = o2.to(torch.device('cuda'))
            r = r.to(torch.device('cuda'))
            d = d.to(torch.device('cuda'))
        q = ac[p_id].q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = ac_targ[p_id].q(o2, ac_targ[p_id].pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        return loss_q, q.detach().cpu().numpy()

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(data, p_id=0):
        o = data['obs']
        if use_gpu:
            o = o.to(torch.device('cuda'))
        q_pi = ac[p_id].q(o, ac[p_id].pi(o))
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = []
    q_optimizer = []
    for i in range(agent_num):
        pi_optimizer.append(Adam(ac[i].pi.parameters(), lr=pi_lr))
        q_optimizer.append(Adam(ac[i].q.parameters(), lr=q_lr))

    # Set up model saving
    #logger.setup_pytorch_saver(ac)
    
    total_steps = steps_per_epoch * epochs

    def update(data, p_id=0):
        # First run one gradient descent step for Q.
        q_optimizer[p_id].zero_grad()
        loss_q, Q_vals = compute_loss_q(data, p_id)
        loss_q.backward()
        q_optimizer[p_id].step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in ac[p_id].q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer[p_id].zero_grad()
        loss_pi = compute_loss_pi(data, p_id)
        loss_pi.backward()
        pi_optimizer[p_id].step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac[p_id].q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac[p_id].parameters(), ac_targ[p_id].parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
        
        return loss_q.item(), loss_pi.item(), Q_vals.mean().item()

    def get_action(o, noise_scale, t=-1):
        action = []
        for i in range(agent_num):
            a = ac[i].act(torch.as_tensor(o, dtype=torch.float32).to(device), use_gpu=use_gpu)
            if t < total_steps / 3 or i == update_policy_id:
                a += noise_scale * np.random.randn(act_dim_sgl)
            action.append(a)
        action = np.concatenate(action, axis=-1)
        return np.clip(action, -act_limit, act_limit)

    def test_agent():
        vals = np.ndarray([num_test_episodes], dtype=np.float64)
        lens = np.ndarray([num_test_episodes], dtype=np.int32)
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset()[0], False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                action = get_action(o, 0)
                o, r, d, _, _ = test_env.step(action)
                ep_ret += r
                ep_len += 1
            vals[j] = ep_ret
            lens[j] = ep_len
            # print(vals[j])
        logger.store(test_avg_r=vals.mean(), test_std_r=vals.std())
        logger.store(test_avg_len=lens.mean())

    # Prepare for interaction with environment
    
    start_time = time.time()
    o, ep_ret, ep_len, loss_q, loss_pi, q_vals, counts =  env.reset(seed=seed)[0], 0, 0, 0, 0, 0, 0
    last_t = 0
    ep_rets = []
    test_env.reset(seed=np.random.randint(100))
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            if args.update_agent > 0:
                a = get_action(o, act_noise, t)
            else:
                a = get_action(o, act_noise)
        else:
            a = np.random.uniform(-act_limit, act_limit, act_dim)

        # Step the env
        o2, r, d, _, info = env.step(a)
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
            ep_rets.append(ep_ret)
            o, ep_ret, ep_len = env.reset()[0], 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
            
                if args.update_agent > 0 and args.reset_buffer and replay_buffer[update_policy_id].size < update_after:
                    break
                    
                for i in range(agent_num):
                    if args.update_agent <= 0:
                        batch = replay_buffer.sample_batch(i, batch_size)
                        lq, lp, qv = update(data=batch, p_id=i)
                        loss_q += lq
                        loss_pi += lp
                        q_vals += qv
                        counts += 1
                    else:
                        batch = replay_buffer.sample_batch(update_policy_id, batch_size)
                        lq, lp, qv = update(data=batch, p_id=update_policy_id)
                        loss_q += lq
                        loss_pi += lp
                        q_vals += qv
                        counts += 1

        # End of epoch handling
        if t >= update_after and (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            train_ret = np.array(ep_rets)
            ep_rets = []
            logger.store(Epoch=epoch)
            logger.store(train_avg_r=train_ret.mean(), train_std_r=train_ret.std())
            test_agent()
            logger.store(loss_Q = loss_q/counts, loss_Pi = loss_pi/counts, Q=q_vals/counts)
            # Test the performance of the deterministic version of the agent.
            loss_q, loss_pi, q_vals, counts = 0, 0, 0, 0
        
            # Log info about epoch
            logger.logging()
        
        
        if args.update_agent > 0 and (t+1-last_t) // args.update_agent >= 1:
            last_t = t+1
            update_policy_id += 1
            if update_policy_id == agent_num:
                update_policy_id = 0
                if args.inc_update and args.update_agent <= 100000:
                    args.update_agent *= 1.04
                    print('change policy:', args.update_agent)
            # if args.reset_buffer and t > total_steps / 3 and args.update_agent > 30000:
            #     replay_buffer[update_policy_id].ptr = 0                
            #     replay_buffer[update_policy_id].size = 0
            #     update_after = 2000
                
            
            

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v4')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    
    #env
    parser.add_argument('--scenario_name', type=str, default='HalfCheetah-v2')
    parser.add_argument('--agent_conf', type=str, default='2x3')
    parser.add_argument('--agent_obsk', type=int, default=2)
    parser.add_argument('--num_agents', type=int, default=2)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--update_agent', type=int, default=1)
    parser.add_argument('--inc_update', default=False, action='store_true')
    parser.add_argument('--reset_buffer', default=False, action='store_true')
    parser.add_argument('--reset_opt', default=False, action='store_true')
    

    
    args = parser.parse_args()
    assert int(args.agent_conf[0]) == args.num_agents
    logger_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

    ddpg(args, lambda : gym.make(args.env), args.env, actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         replay_size=args.buffer_size, logger_dir=logger_dir, model_name=args.exp_name)
         
