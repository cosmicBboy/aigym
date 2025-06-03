# Project Roadmap

- ✅ Create web environment of Wikipedia pages
- ✅ Create generic agent to traverse Wikipedia pages
- ✅ Create example Ollama-based agent
- ✅ Create Openai- and Gemini-based LLM agent
- ✅ Observations should be web page chunks
- ✅ Look into breaking down agent prompts into multiple steps: perception and action
- ✅ Env should select a start url and find a target page via random walk n pages away
- ✅ Spec out the RL training procedure
- ☑️ Reward signal: -1 if not if current state is not at target
- ☑️ Handle cases where action url are not valid links
- ☑️ Implement single observation -> action -> reward loop
- ☑️ Collect trajectories through the web env graph
- ☑️ Implement offline policy update on an LLM using `trl` library
- ☑️ Measure task performance as cumulative reward
- ☑️ Measure performance against common LLM benchmarks
- ☑️ Implement a wikipedia-based environment using https://huggingface.co/datasets/wikimedia/structured-wikipedia

## Training Procedure

The RL training procedure for multi-step episodes is slightly different from
the common math and coding RLVR settings, where the agent receives
one observation, produces an action (composed of a sequence of "micro actions"
containing a thinking trace and the final answer), and receive a reward. This
constitutes a single step in a multi-step episodic setting.

At the start of a multi-step episodic setting, the agent receives an observation,
takes an action, and then receives a reward and subsequent observation. The
agent then takes `n` actions until it reaches a terminal state. Returns
must be broadcast appropriately so that the agent policy can be updated for
actions earlier in the trajectory.

## Environment

In a general sense, the RL environment is represented as a graph of nodes, where
each node is an observation, and edges between nodes represent some causal
relationship such `environment(observation, action) -> (next_observation, reward)`.

In the specific case of the **Wikipedia Maze** environment, each node is a
wikipedia page, and nodes are connected via directed links from one page to
another. The training episode is initialized by selecting a random wikipedia page
and doing `n` random hops sampled from links on the current page. The agent's
objective is to produce thinking traces and actions that get it from the
initial random page to the target page `n` hops away.

Going back to a more general framing, this training regime is important for
reinforcement fine-tuning problems that can be cast as a graph of nodes, where:

- Edges are traversible via some action output that can be interpreted by the
  environment to produce the reward and following observation (e.g. `json`
  strings containing function calls to tools).
- Observations contain information about the environment, such as search engine
  results, code sandbox outputs, etc. and can be appended to the agent's context.
- Rewards are defined by either deterministic reward functions (e.g. `is_target_url(observation)`)
  or fuzzy reward functions that use perhaps other LLMs to rate the output
  according to some criteria.

The training procedure consists of two phases:

### Generate trajectories phase

- For each completion episode:
  - Generate `g` completion trajectories per episode
    - Collect `n` `(observation, action, reward)` tuples
    - Rewards are determined per step:
      - -0.1 per step
      - 1 if observation is in sampled trajectory
      - 10 if observation is the target page
    - Summer per-step rewards produces the total reward for the trajectory.
  - Compute the advantage for the completion trajectories:
    - Option 1: `(x - mean) / std` (normalized)
    - Option 2: or `x - mean` (unnormalized)
  - Propagate advantages to prior actions in each trajectory
    using a `gamma` discount factor. For more information on how to implement
    this, see [here](https://www.perplexity.ai/search/8c78841f-4205-478a-8181-aaa21b74ac75).
  - Save completions in experience replay buffer

### Update agent policy phase

- For each completion batch in the experience replay buffer:
  - Compute GRPO loss:
    - Proximal-policy-adjusted advantage with clipping
    - KL divergance approximation
  - (Optional): Accumulate gradients for each trajectory. This may bias the
    results in unexpected ways due to data dependence of the observations,
  - Update policy weights

> [!NOTE]
> An alternative update policy procedure would be to perform updates per step,
> i.e. instead of batching the data points by completion trajectory such that
> each batch is `(n_trajectories * n_steps, n_tokens, embedding_dim)`, each
> data point is a randomly sampled step in the whole dataset
> `(n_steps, n_tokens, embedding_dim)`, which treats this dataset as IID.
