# Group 22

- Andrew Zhang (andrew_zhang@college.harvard.edu)
- Diwei Zhang (diwei_zhang@hsph.harvard.edu)
- Lotus Xia (lxia@g.harvard.edu)
- Neil Sehgal (neil_sehgal@g.harvard.edu)

# SALAD: An AutoDiff Package

This project is available for download through 
```
pip install cs107-salad --extra-index-url=https://test.pypi.org/simple/
```
(Will change this once we upload to main PyPi)

Documentation available [here](https://duckduckgo.com) (Link not correct)

PyPi link [here](https://test.pypi.org/project/cs107-salad/) (Link needs to be updated for main PyPi)

## Broader Impact

Our extension involves several optimization methods: gradient descent, stochastic gradient descent, BFGS, and Newton’s method. By itself, these are facially neutral algorithms – they are not discriminatory on their face – yet they have clear potential discriminatory applications and effects. Our subpackages could easily be extended and combined with other software modules to create bad, biased, or unethical AI models. For instance, Amazon was forced to scrap its automated recruiting tool when it was revealed that the algorithm was biased against women. The algorithm was biased because it was trained on resumes previously submitted to Amazon. However, reflecting unequal trends pervasive throughout the tech industry, most candidates that made it through the recruitment process were men. Amazon’s model, when optimized to this data, learned that resumes from men were preferable to resumes from women. One could imagine how our package, when used to optimize on biased data from any setting would have negative disparate impacts.



## Software Inclusivity

The salad package warmly welcomes all people regardless of background to contribute to the development of our package. Our package is open-source, easily accessible through Github and PyPi, and licensed under the MIT License. Nonetheless, we recognize there are many structural barriers for the average person to interact with and contribute to our software package. For instance, while we aimed to have high code quality with clear comments, our comments are all in English. As a result, non-native English speakers are at a disadvantage in contributing to our package. Moreover, computer science/programming education and education in general are not shared equally across society. BIPOC, the poor, women, and other minorities are all not afforded the same educational opportunities as the cis-male White and Asian men that currently make up the majority of programmers in the US. Such underrepresented minorities face more constraints in contributing to open source software projects like ours. 
 

