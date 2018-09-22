# Temporal Difference - SARSA/Q-Learning
##### For Assignments COMP532 Machine Learning and BioInspired Optimisation, MSc Computer Science, University of Liverpool 
Python implementation to compare the optimal results found usign a SARSA/Q-Learning approch to the
cliff walking game, presented in [Sutton, Richard S., and Andrew G. Barto. *Reinforcement learning: An introduction*. MIT press, 1998](http://incompleteideas.net/book/bookdraft2017nov5.pdf), Figure 6.4.

With reinforcement leraning we face the problem of exploration and exploitation where this approaches fall to two classes: on-policy and off-policy.
___
#### SARSA: On-policy TD Control
First, learn action-value function rather than a state-value function. For an on-policy method, we estimate Q^π (s,a) for the current behaviour policy π and for all states s and actions a. 
We consider transitions from state-action pair to state-action pairs and learn values of these pairs. 
Formally, there are identical, they are both Markov chains with rewards. 
The theorems assuring the convergence of state values under TD(0) also apply to the corresponding algorithm:

Q(s_t,a_t )←Q(s_t,a_t )+α[r_(t+1)+γQ(s_(t+1),a_(t+1) )-Q(s_t,a_t)]

Update is done after every transition from a non terminal state s_t, if s_(t+1) is terminal, then Q(s_(t+1),a_(t+1)) is defined as zero.
This rule uses every element of the quintuple of events, (s_t,a_t,r_(t+1),s_(t+1),a_(t+1)), making up the transition from one state-action to the next. 
Called the SARSA algorithm.
We continually estimate Q^π for the behaviour policy π, and at the same time, change π toward a greediness with respect to Q^π. The convergence properties of the SARSA algorithm depend on the nature of the policy’s dependence on Q.

```
Initialize Q(s,a),∀ s∈S,a∈A(s) arbitrarily, and Q(terminal state,∙) = 0
Repeat (for each episode):
     Initialize s
     Choose a from s using policy derived from Q (e.g., ε-greedy)
     Repeat (for each step of episode):
          Take action a, observe r,r'
          Choose a' from s' using policy derived from Q (e.g., ε-greedy)
          Q(s_t,a_t )←Q(s_t,a_t )+α[r_(t+1)+γQ(s_(t+1),a_(t+1) )-Q(s_t,a_t)]
          s←s^';a←a'
     until S is terminal
```
One could use ε-greedy or ε-soft policies, Sarsa converges with probability 1 to an optimal policy and action-value function as long as all state-action pairs are visited an infinite number of times and the policy converges in the limit to the greedy policy.
If a policy was never found that caused the agent to stay in the same state, then the next episode would never end. Step-by-step learning methods such as SARSA do not have this problem because they quickly learn during the episode that such polices are poor.


___
#### Q-Learning: Off-policy TD control
Off-policy TD control algorithm known as Q-learning:

Q(s_t,a_t )←Q(s_t,a_t )+α[r_(t+1)+γ  max┬a⁡〖Q(s_(t+1),a)-Q(s_t,a_t )〗 ]

The learned action-value function Q directly approximates Q^*, the optimal action-value function, independent of the policy being followed. This simplifies the analysis of the algorithm and enabled early convergence. The policy still has an effect in that it determines which state-action pairs are visited and updated. However, the only requirements for convergence is that all pairs continue to be updated. 
This is a minimal requirement in the sense that any method guaranteed to find optimal behaviour in the general case must require it. Under this assumption and a variant of the usual stochastic approximation conditions on the sequence of step-size parameters, Q has been shown to converge with probability 1 to Q^*.
The rule updates a state-action pair, so the top node (root of the update), must be small, filled action node. Update is also from action nodes, maximizing over all those actions possible in the next state. Thus the bottom nodes of the backup diagram should be all action nodes. 
```
Initialize Q(s,a),∀ s∈S,a∈A(s) arbitrarily, and Q(terminal state,∙) = 0
Repeat (for each episode):
     Initialize s
     Repeat (for each step of episode):
          Choose a from s using policy derived from Q (e.g., ε-greedy)
          Take action a, observe r,r'
          Q(s_t,a_t )←Q(s_t,a_t )+α[r_(t+1)+γ  max┬a⁡Q(s_(t+1),a_t )-Q(s_t,a_t)]
          s←s^';a←a'
     until S is terminal
```
