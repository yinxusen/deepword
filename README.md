# deeptextworld

Create auto-agent to play text-based games

Our testbed is [TextWorld](https://github.com/microsoft/TextWorld). Our BaseAgent is built upon TextWorld. However, our agent core - the most important model part - is independent with specific game frameworks or games.

The BaseAgent works with specific games / frameworks, fetching important information from games, and gives actions as feedback.

The BaseCore works independently, requires only trajectory and actions, etc.

We support

1. DQN agents

2. DRRN agents

3. generation agents

4. Bert commonsense agents

We provide

1. CNN models

2. Transformer models

3. Pointer-generator models for generation agents

cite our papers

```
@article{DBLP:journals/corr/abs-1905-02265,
  author    = {Xusen Yin and
               Jonathan May},
  title     = {Comprehensible Context-driven Text Game Playing},
  journal   = {CoRR},
  volume    = {abs/1905.02265},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.02265},
  archivePrefix = {arXiv},
  eprint    = {1905.02265},
  timestamp = {Mon, 27 May 2019 13:15:00 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1905-02265},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

@misc{yin2019learn,
    title={Learn How to Cook a New Recipe in a New House: Using Map Familiarization, Curriculum Learning, and Bandit Feedback to Learn Families of Text-Based Adventure Games},
    author={Xusen Yin and Jonathan May},
    year={2019},
    eprint={1908.04777},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```


