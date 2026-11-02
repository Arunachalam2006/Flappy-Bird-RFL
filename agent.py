import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import torch
import itertools
import yaml
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random


if torch.backends.mps.is_available(): # checks if gpu is available for faster computation
    device="mps"
elif torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"


RUNS_DIR="runs"
os.makedirs(RUNS_DIR,exist_ok=True)


class Agent:
    def __init__(self,param_set):
        self.param_set=param_set


        with open("parameters.yaml","r") as f:
            all_param_set=yaml.safe_load(f)
            params=all_param_set[param_set]

        self.alpha=params["alpha"]
        self.gamma=params["gamma"]

        self.epsilon_min=params["epsilon_min"]
        self.reward_threshold=params["reward_threshold"]
        self.network_syn_rate=params["network_sync_rate"]

        self.epsilon_init=params["epsilon_init"]
        self.epsilon_decay=params["epsilon_decay"]
        self.replay_memory_size=params["replay_memory_size"]
        self.loss_fn=nn.MSELoss()
        self.optimizer=None
        self.mini_batch_size=params["mini_batch_size"]

        self.LOG_FILE=os.path.join(RUNS_DIR,f"{self.param_set}.log")
        self.MODEL_FILE=os.path.join(RUNS_DIR,f"{self.param_set}.pt")

    def run(self,is_training=True,render=False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)

        num_states=env.observation_space.shape[0]  # input dim
        num_actions=env.action_space.n  # output dim

        policy_dqn=DQN(num_states,num_actions).to(device)


        if is_training:
            memory=ReplayMemory(self.replay_memory_size)
            epsilon=self.epsilon_init

            target_dqn=DQN(num_states,num_actions).to(device)
            # resume from an earlier model if the file exists
            # set resume_file to a .pt file name to continue from an older model (it must have the same input size)
            resume_file=None
            if resume_file is not None and os.path.exists(resume_file):
                policy_dqn.load_state_dict(torch.load(resume_file,map_location=device))

            # copy the wt & bias vals from policy => target
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps=0

            self.optimizer=optim.Adam(policy_dqn.parameters(),lr=self.alpha)

            best_reward=float("-inf")

            reward_history=[]
            best_avg=float("-inf")

        else:
            # best policy load
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count():
            state,_=env.reset()
            state=torch.tensor(state,dtype=torch.float,device=device)

            episode_reward=0
            done=False
            while (not done and episode_reward < self.reward_threshold):
                    if is_training and random.random() < epsilon:
                        action = torch.tensor(env.action_space.sample(), dtype=torch.long, device=device) # explore

                    else:
                        with torch.no_grad():
                            action=policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() # exploit
                    # Processing:terminated => done
                    next_state, reward, terminated, truncated, _ = env.step(action.item())
                    done = terminated or truncated
                    episode_reward+=reward

                    # create tensors    
                    reward=torch.tensor(reward,dtype=torch.float,device=device)
                    next_state=torch.tensor(next_state,dtype=torch.float,device=device)
             
                    if is_training:
                        memory.append((state,action,next_state,reward,done))    
                        steps+=1

                    if is_training and len(memory)>self.mini_batch_size:
                        # get sample
                        mini_batch=memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch,policy_dqn,target_dqn)

                        # sync the networks
                        if steps > self.network_syn_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            steps=0


                    state=next_state


            if is_training:
                print(f"episode={episode+1} with total reward={episode_reward} & epsilon={epsilon}")
            else:
                print(f"episode={episode+1} with total reward={episode_reward}")
            if is_training:
                # epsilon decay
                epsilon=max(epsilon * self.epsilon_decay,self.epsilon_min)

                if episode_reward>best_reward:
                    log_msg=f"best reward={episode_reward} for episode={episode+1}"

                    with open(self.LOG_FILE,"a") as f:
                        f.write(log_msg + "\n")

                    best_reward=episode_reward

                # save the model by the average of the last 100 episodes, not by one lucky episode
                reward_history.append(episode_reward)
                if len(reward_history)>=100:
                    avg_reward=sum(reward_history[-100:])/100
                    if avg_reward>best_avg:
                        best_avg=avg_reward
                        torch.save(policy_dqn.state_dict(),self.MODEL_FILE)
                        with open(self.LOG_FILE,"a") as f:
                            f.write(f"best avg of last 100 episodes={avg_reward} at episode={episode+1}\n")
                    if (episode+1)%100==0:
                        print(f"avg of last 100 episodes={avg_reward} & best avg so far={best_avg}")

            # env.close()-manually stop the training process


    def optimize(self, mini_batch, policy_dqn, target_dqn):
    # get experinces => batch train
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        # calculate target Q-values - if terminations=true, zero
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        # calculate y_pred i.e. Q-value from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # compute loss
        loss = self.loss_fn(current_q, target_q)

        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("hyperparameters", help="")
    parser.add_argument("--train", help="Training mode", action="store_true")
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)