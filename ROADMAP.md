# Project Roadmap

- ✅ Create web environment of Wikipedia pages
- ✅ Create generic agent to traverse Wikipedia pages
- ✅ Create example Ollama-based agent
- ✅ Create Openai- and Gemini-based LLM agent
- ✅ Observations should be web page chunks
- ✅ Look into breaking down agent prompts into multiple steps: perception and action
- ✅ Env should select a start url and find a target page via random walk n pages away
- ☑️ Reward signal: -1 if not if current state is not at target
- ☑️ Handle cases where action url are not valid links
- ☑️ Implement single observation -> action -> reward loop
- ☑️ Collect trajectories through the web env graph
- ☑️ Implement offline policy update on an LLM using `trl` library
- ☑️ Measure task performance as cumulative reward
- ☑️ Measure performance against common LLM benchmarks
- ☑️ Implement a wikipedia-based environment using https://huggingface.co/datasets/wikimedia/structured-wikipedia
