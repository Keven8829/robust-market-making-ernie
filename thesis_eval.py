import gymnasium as gym
from env import MarketEnv
from agent import MAPPOAgent, ERNIE_PPO_Agent
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

def estimate_lipschitz_constant(agent, env, num_samples=1000, epsilon=0.01):
    """
    Estimates the local Lipschitz constant of the Agent's policy.
    L ~ || pi(s) - pi(s') || / || s - s' ||
    """
    lipschitz_values = []
    
    for _ in range(num_samples):
        # Sample random state from Environment space (approximate)
        # We assume independent sampling or just take from a rollout
        # Let's simple create random states within reasonable bounds
        # Stats: Mid=100, Spread=0.01-0.05, Imbalance=-1to1, RSI=0-100, MACD=-2to2, Inv=-10to10
        state = env.observation_space.sample() 
        # But gym sample often gives uniform noise, data might be specific.
        # Better to grab from a random reset
        s, _ = env.reset()
        state = torch.FloatTensor(s)
        
        # Perturb state
        noise = torch.randn_like(state) * epsilon
        state_prime = state + noise
        
        # Get Action Distributions
        with torch.no_grad():
            logits = agent.actor(state.unsqueeze(0))
            logits_prime = agent.actor(state_prime.unsqueeze(0))
            
            probs = F.softmax(logits, dim=-1)
            probs_prime = F.softmax(logits_prime, dim=-1)
            
        # Distance actions (Total Variation or L2 of probs)
        # ERNIE paper often uses KL, but for Lipschitz constant L2 is standard
        dist_pi = torch.norm(probs - probs_prime, p=2).item()
        dist_s = torch.norm(noise, p=2).item()
        
        if dist_s > 1e-9:
            L = dist_pi / dist_s
            lipschitz_values.append(L)
            
    return np.mean(lipschitz_values), np.std(lipschitz_values)

def calculate_risk_metrics(returns):
    """
    Calculates financial risk metrics from a series of PnL/Returns.
    """
    returns = np.array(returns)
    if len(returns) == 0: return {}
    
    # Cumulative PnL
    total_return = np.sum(returns)
    
    # Sharpe Ratio (assuming risk-free rate = 0 for simplicity, daily-ish steps)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns) + 1e-9
    sharpe = mean_ret / std_ret
    
    # Sortino (Downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) + 1e-9 if len(downside_returns) > 0 else 1.0
    sortino = mean_ret / downside_std
    
    # Max Drawdown
    cum_returns = np.cumsum(returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (peak - cum_returns)
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # VaR (Value at Risk) 95%
    var_95 = np.percentile(returns, 5)
    
    return {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown,
        "VaR (95%)": var_95
    }

def run_thesis_evaluation():
    # Load Models
    env = MarketEnv(ticker='GC=F')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    baseline = MAPPOAgent(state_dim, action_dim)
    ernie = ERNIE_PPO_Agent(state_dim, action_dim)
    
    try:
        baseline.actor.load_state_dict(torch.load('models/baseline_actor.pth'))
        ernie.actor.load_state_dict(torch.load('models/ernie_actor.pth'))
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}. Ensure training has run.")
        return

    # 1. Theoretical: Lipschitz Constant
    print("Evaluating Lipschitz Constants...")
    lip_baseline_mean, lip_baseline_std = estimate_lipschitz_constant(baseline, env)
    lip_ernie_mean, lip_ernie_std = estimate_lipschitz_constant(ernie, env)
    
    print(f"Baseline Lipschitz: {lip_baseline_mean:.4f} +/- {lip_baseline_std:.4f}")
    print(f"ERNIE Lipschitz:    {lip_ernie_mean:.4f} +/- {lip_ernie_std:.4f}")
    
    # 2. Financial: Regime Shift (Bear Market)
    # We force a bear market by modifying the env data generator temporarily or just injecting a downward trend
    # Or we use Scenario B (Noisy) which is a proxy for volatility
    # Let's do a run on standard env first
    
    def run_episode(agent, env):
        state, _ = env.reset()
        done = False
        rewards = []
        while not done:
            action, _ = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            state = next_state
        return rewards

    print("Running Risk Evaluation episodes...")
    baseline_rewards = run_episode(baseline, env)
    ernie_rewards = run_episode(ernie, env)
    
    metrics_baseline = calculate_risk_metrics(baseline_rewards)
    metrics_ernie = calculate_risk_metrics(ernie_rewards)
    
    # DataFrame for Table
    df_metrics = pd.DataFrame([metrics_baseline, metrics_ernie], index=['Baseline', 'ERNIE'])
    print("\nRisk Metrics:")
    print(df_metrics)
    
    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Lipschitz
    agents = ['Baseline', 'ERNIE']
    means = [lip_baseline_mean, lip_ernie_mean]
    errors = [lip_baseline_std, lip_ernie_std]
    
    ax[0].bar(agents, means, yerr=errors, capsize=10, color=['red', 'green'], alpha=0.7)
    ax[0].set_title('Policy Smoothness (Lipschitz Constant)')
    ax[0].set_ylabel('Lipschitz Constant (Lower is Better)')
    
    # Plot 2: Cumulative Returns
    ax[1].plot(np.cumsum(baseline_rewards), label='Baseline', color='red', alpha=0.7)
    ax[1].plot(np.cumsum(ernie_rewards), label='ERNIE', color='green', alpha=0.7)
    ax[1].set_title('Cumulative PnL (Eval Episode)')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig('thesis_results.png')
    print("Saved thesis_results.png")
    
    # Save Risk Table
    with open('risk_table.md', 'w') as f:
        f.write(df_metrics.to_markdown())
    print("Saved risk_table.md")

if __name__ == '__main__':
    run_thesis_evaluation()
