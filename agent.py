import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn import functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)

class MAPPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # Memory
        self.buffer = []

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state)
            dist = Categorical(probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        
        return action.item(), action_logprob.item()

    def store_transition(self, transition):
        # transition: (state, action, logprob, reward, done)
        self.buffer.append(transition)

    def update(self):
        if not self.buffer:
            return

        # Unpack buffer
        states, actions, logprobs, rewards, dones = zip(*self.buffer)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_logprobs = torch.FloatTensor(logprobs)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # Monte Carlo estimate of returns
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
            
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7) # Normalize

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            probs = self.actor(states)
            dist = Categorical(probs)
            new_logprobs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
            state_values = self.critic(states).squeeze()
            
            # Ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(new_logprobs - old_logprobs)

            # Surrogate Loss
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Loss
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values, returns)
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()

            # Take gradient step
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()

        self.buffer = [] # Clear memory


class ERNIE_PPO_Agent(MAPPOAgent):
    """
    ERNIE Agent: Extends MAPPO with Adversarial Regularization (Stackelberg Game).
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4,
                 epsilon=0.1, eta=0.01, vigilance=0.1, pgd_steps=10):
        super().__init__(state_dim, action_dim, lr, gamma, eps_clip, k_epochs)
        self.epsilon = epsilon   # Perturbation bound (e.g. L2 norm)
        self.eta = eta           # Step size for PGD
        self.vigilance = vigilance # Lambda (Robustness weight)
        self.pgd_steps = pgd_steps

    def compute_adversarial_perturbation(self, states):
        """
        Stackelberg Follower Step: Find delta that maximizes KL(pi(s+delta) || pi(s))
        """
        # Detach states to prevent gradient flow to actor during attack optimization
        states = states.detach()
        delta = torch.zeros_like(states, requires_grad=True)
        
        # Target distribution (Clean)
        with torch.no_grad():
            clean_probs = self.actor(states)
            
        # PGD Loop
        for _ in range(self.pgd_steps):
            perturbed_probs = self.actor(states + delta)
            
            # KL Divergence: KL(Clean || Perturbed) or KL(Perturbed || Clean)?
            # PRD: Maximize KL(pi(s+delta) || pi(s)) which means how different perturbed is from clean.
            # Usually we want the agent to be smooth, so we minimize KL. 
            # The ADVERSARY wants to MAXIMIZE KL (find worst state).
            # The AGENT wants to MINIMIZE KL (be robust).
            
            # Note: PyTorch kl_divergence(p, q) is sum(p * log(p/q))
            # KL(Clean || Perturbed)
            
            # Convert to distributions
            dist_clean = Categorical(clean_probs)
            dist_perturbed = Categorical(perturbed_probs)
            
            kl = torch.distributions.kl.kl_divergence(dist_clean, dist_perturbed).mean()
            
            # Gradient Ascent on KL (Adversary maximizes divergence)
            kl.backward()
            
            with torch.no_grad():
                delta.data += self.eta * delta.grad.sign()
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
                delta.grad.zero_()
                
        return delta.detach()

    def update(self):
        if not self.buffer:
            return

        # Unpack buffer
        states, actions, logprobs, rewards, dones = zip(*self.buffer)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_logprobs = torch.FloatTensor(logprobs)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # Monte Carlo estimate of returns
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
            
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Compute Adversarial Perturbation (Stackelberg Attack)
            delta = self.compute_adversarial_perturbation(states)
            perturbed_states = states + delta
            
            # Evaluating old actions and values
            probs = self.actor(states)
            dist = Categorical(probs)
            new_logprobs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
            state_values = self.critic(states).squeeze()
            
            # Ratio
            ratios = torch.exp(new_logprobs - old_logprobs)

            # Surrogate Loss
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # PPO Loss
            ppo_loss = -torch.min(surr1, surr2).mean() + \
                       0.5 * nn.MSELoss()(state_values, returns) - \
                       0.01 * dist_entropy.mean()
                       
            # Robustness Regularization (Stackelberg Defense)
            # Minimize KL(Clean || Perturbed)
            dist_clean = Categorical(probs)
            dist_perturbed = Categorical(self.actor(perturbed_states))
            kl_loss = torch.distributions.kl.kl_divergence(dist_clean, dist_perturbed).mean()
            
            total_loss = ppo_loss + self.vigilance * kl_loss

            # Take gradient step
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            total_loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()

        self.buffer = []
