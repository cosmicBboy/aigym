# Project TODOs

- [X] create basic ollama-based LLM agent
- [X] observations should be web page chunks
- [X] look into breaking down agent prompts into multiple steps
- [ ] env should select a target page and random walk n pages away
- [ ] reward signal: -1 if not if current state is not at target
- [ ] handle cases where action url are not valid links
- [ ] implement single observation -> action -> reward loop
- [ ] collect trajectories through the web env graph
- [ ] implement offline policy update on an LLM using `trl` library
- [ ] measure task performance as cumulative reward
- [ ] measure performance against common LLM benchmarks
- [ ] implement a wikipedia-based environment using https://huggingface.co/datasets/wikimedia/structured-wikipedia
