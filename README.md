```text
██████╗ ███████╗███████╗██████╗ ██╗    ██╗ ██████╗ ██████╗ ██████╗
██╔══██╗██╔════╝██╔════╝██╔══██╗██║    ██║██╔═══██╗██╔══██╗██╔══██╗
██║  ██║█████╗  █████╗  ██████╔╝██║ █╗ ██║██║   ██║██████╔╝██║  ██║
██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██║███╗██║██║   ██║██╔══██╗██║  ██║
██████╔╝███████╗███████╗██║     ╚███╔███╔╝╚██████╔╝██║  ██║██████╔╝
╚═════╝ ╚══════╝╚══════╝╚═╝      ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═════╝ 
```

# DeepWord

Create auto-agent to play text-based games

Our testbed is [TextWorld](https://github.com/microsoft/TextWorld).
Our BaseAgent is built upon TextWorld.
However, our agent core is independent with specific game frameworks or games.

The BaseAgent works with specific games/frameworks, fetching important information from games, and gives actions as feedback.

The BaseCore works independently, requiring only trajectory and actions, etc.

Read the [tutorial.md](tutorial.md) for details.

Cite our papers

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

@misc{yin2020zeroshot,
    title={Zero-Shot Learning of Text Adventure Games with Sentence-Level Semantics},
    author={Xusen Yin and Jonathan May},
    year={2020},
    eprint={2004.02986},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

@misc{yin2020learning,
      title={Learning to Generalize for Sequential Decision Making}, 
      author={Xusen Yin and Ralph Weischedel and Jonathan May},
      year={2020},
      eprint={2010.02229},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
