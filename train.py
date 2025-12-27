import gymnasium as gym
from env import MarketEnv
from agent import MAPPOAgent, ERNIE_PPO_Agent
import numpy as np
import torch
import os
import argparse

def train(args):
    env = MarketEnv(ticker='GC=F', start_date='2023-01-01', end_date='2023-06-01')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if args.agent == 'baseline':
        agent = MAPPOAgent(state_dim, action_dim, lr=0.0003, gamma=0.99, k_epochs=5)
    elif args.agent == 'ernie':
        agent = ERNIE_PPO_Agent(state_dim, action_dim, lr=0.0003, gamma=0.99, k_epochs=5,
                                epsilon=0.05, eta=0.01, vigilance=0.1)
    else:
        raise ValueError("Invalid agent type. Use 'baseline' or 'ernie'.")
    
    max_episodes = args.episodes
    max_timesteps = 1000 
    update_timestep = 2000 
    timestep = 0
    
    print(f"Starting training with {args.agent} agent for {max_episodes} episodes.")
    
    for i_episode in range(1, max_episodes+1):
        state, _ = env.reset()
        ep_reward = 0
        
        for t in range(max_timesteps):
            timestep += 1
            
            # Select action
            action, logprob = agent.get_action(state)
            
            # Step
            next_state, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            
            # Save data
            agent.store_transition((state, action, logprob, reward, done))
            
            state = next_state
            
            # Update Agent
            # Note: ERNIE update includes Stackelberg logic internally
            if timestep % update_timestep == 0:
                agent.update()
                
            if done or truncated:
                break
                
        if i_episode % 10 == 0:
            print(f"Episode {i_episode}\tAverage Reward: {ep_reward:.2f}\tInventory: {info['inventory']}\tWealth: {info['wealth']:.2f}")

    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(agent.actor.state_dict(), f'models/{args.agent}_actor.pth')
    torch.save(agent.critic.state_dict(), f'models/{args.agent}_critic.pth')
    print(f"Training finished. Models saved to models/{args.agent}_*.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='baseline', help='baseline or ernie')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes')
    args = parser.parse_args()
    train(args)
