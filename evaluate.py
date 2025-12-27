import gymnasium as gym
from env import MarketEnv
from agent import MAPPOAgent, ERNIE_PPO_Agent
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

def evaluate(agent_type='baseline', model_path=None, noise_std=0.0, shock_prob=0.0):
    env = MarketEnv(ticker='GC=F', start_date='2023-01-01', end_date='2023-06-01')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if agent_type == 'baseline':
        agent = MAPPOAgent(state_dim, action_dim)
        path = model_path or 'models/baseline_actor.pth'
    else:
        agent = ERNIE_PPO_Agent(state_dim, action_dim)
        path = model_path or 'models/ernie_actor.pth'
        
    try:
        agent.actor.load_state_dict(torch.load(path))
        print(f"Loaded model from {path}")
    except FileNotFoundError:
        print(f"Model not found at {path}, returning random agent performance or 0.")
        # If model doesn't exist, we can't eval. But for flow, let's continue with untrained
        pass
    
    state, _ = env.reset()
    done = False
    
    rewards = []
    
    max_steps = 1000
    for t in range(max_steps):
        # Inject Gaussian Noise (Scenario B)
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=state.shape)
            state = state + noise
            
        # Inject Shock (Scenario C) - Simulate Flash Crash on Price (index 0)
        # If shock happens, Mid Price drops significantly and Spread widens
        if shock_prob > 0 and np.random.rand() < shock_prob:
             # Shock: Price drops 5%, Spread jumps 10x
             # State indices: 0=Mid, 1=Spread
             state[0] = state[0] * 0.95 
             state[1] = state[1] * 10.0
        
        action, _ = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        
        rewards.append(reward)
        state = next_state
        
        if done:
            break
            
    return np.cumsum(rewards)

def run_evaluation(episodes=10):
    # Scenario A: Clean
    baseline_clean = evaluate('baseline', noise_std=0.0)
    ernie_clean = evaluate('ernie', noise_std=0.0)
    
    # Scenario B: Noisy
    noise_level = 2.0 # Standard dev of noise
    baseline_noisy = evaluate('baseline', noise_std=noise_level)
    ernie_noisy = evaluate('ernie', noise_std=noise_level)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Scenario A: Clean Market")
    plt.plot(baseline_clean, label='Baseline (Clean)', alpha=0.7)
    plt.plot(ernie_clean, label='ERNIE (Clean)', alpha=0.7)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Cumulative Reward (PnL)")
    
    plt.subplot(1, 2, 2)
    plt.title(f"Scenario B: Noisy Market (std={noise_level})")
    plt.plot(baseline_noisy, label='Baseline (Noisy)', alpha=0.7, linestyle='--')
    plt.plot(ernie_noisy, label='ERNIE (Noisy)', alpha=0.7)
    plt.legend()
    plt.xlabel("Time")
    
    plt.tight_layout()
    plt.savefig('results.png')
    print("Results saved to results.png")

if __name__ == '__main__':
    run_evaluation()
