# What is Raphael
Raphael is a general-purpose, scalable, continuously learning neural network or brain.

## What is difference
Neural networks represented by CNNs, RNNs, and Transformers rely on large amounts of data and computational resources and have strong representation learning capabilities, but they are often static, with fixed parameters after training. Although Transformers, led by GPT, have expanded their number of neurons (parameters) to approach the number of neurons in the human brain, resulting in unexpected emergent effects, it is likely that these models will continue to increase their parameters in the future.

However, fundamentally, although these models have addressed a key issue: the importance of the number of neurons and their interactions, they have not considered the redundancy of parameters and the necessity of dynamically adjusting them. Dynamically increasing or decreasing parameters during model operation will be an important direction.

This method (referring to the dynamic addition and reduction of neurons) can truly simulate the lifecycle of neurons and address a problem that mathematics alone cannot solve: the loss function becomes fixed after fitting, causing the model to lose its ability to learn. I have read numerous solutions aimed at preventing the training loss function from becoming fixed, all of which approach the problem from a mathematical perspective. They seek to use a function that is continuous, smooth, and differentiable everywhere to prevent the loss function from becoming fixed through mathematical operations. However, I believe we should consider this issue from the objective laws of biology: ~~our brain continuously replaces neurons (metabolism).~~ If we apply this mechanism to neural networks, rather than relying solely on mathematical approaches, could we solve the problem of the loss function becoming fixed?

> Notify:  
> The number of neurons in the brain tends to stabilize after adulthood, so the statement that 'our brains are replacing neurons all the time' is a wrong description.  
> Thanks my friend 福酱(18963870758@163.com).

## Explanation
Here is an explanation of why using the interaction of mechanisms instead of purely mathematical methods. We know that individual physical or chemical principles are simple: for example, the entropy principle and the enthalpy principle. However, the combination of these two principles forms the principle of minimum free energy, which explains how neurons minimize energy to predict events. This combination can also give rise to various anthropological and sociological phenomena, such as human curiosity about the unknown (consistent with the principle of entropy increase in information theory), scientific induction of new knowledge into existing systems (reducing cognitive load), and laziness in states of abundance or extreme scarcity (a self-protection mechanism of biological systems).

From this perspective, we can see that the combination of physical and chemical principles can produce a series of real-world phenomena. Additionally, the term "emergence" refers to the regular state achieved by the interaction of various mechanisms in an ecosystem. In games like The Legend of Zelda: Breath of the Wild, Dwarf Fortress, similar interaction mechanisms present situations more in line with natural conditions. Emergent phenomena also align with the study of chaos theory.

In summary, sometimes 1+1 is indeed not equal to 2. The interaction between them is greater than 2, although the effect is small and unquantifiable.

## Cell division(focal point)
- [ ] Learn the biological theories related to cell division.

- [ ] Continue to write about Raphael’s 'principles'.

# How to use
This project uses Anaconda to manage the required packages,please make sure you have the relevant Anaconda environment.
## Step
1. Prepare Anaconda environment.
2. Use the following command in this project folder to download the packages required for the project.
```bash
conda env create -f ./environment.yml [-p </your/path>]
```

3. Use the following command to activate the "Raphael" environment.
```bash
conda activate </your/env/path>
```
4. If you no longer want to continue developing "Raphael", and do not want to retain any "Raphael" related environment; please use the following commands:

Deactivate command:
```bash
conda deactivate
```

Remove entire environment command:
```bash
conda remove -p </path/to/your/env> --all
```