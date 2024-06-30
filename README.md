# What is Raphael

Raphael is a general-purpose, scalable, continuously learning neural network or brain.

## What is difference

Neural networks represented by CNNs, RNNs, and Transformers rely on large amounts of data and computational resources and have strong representation learning capabilities, but they are often static, with fixed parameters after training. Although Transformers, led by GPT, have expanded their number of neurons (parameters) to approach the number of neurons in the human brain, resulting in unexpected emergent effects, it is likely that these models will continue to increase their parameters in the future.

However, fundamentally, although these models have addressed a key issue: the importance of the number of neurons and their interactions, they have not considered the redundancy of parameters and the necessity of dynamically adjusting them. Dynamically increasing or decreasing parameters during model operation will be an important direction.

This method (referring to the dynamic addition and reduction of neurons) can truly simulate the lifecycle of neurons and address a problem that mathematics alone cannot solve: the loss function becomes fixed after fitting, causing the model to lose its ability to learn. I have read numerous solutions aimed at preventing the training loss function from becoming fixed, all of which approach the problem from a mathematical perspective. They seek to use a function that is continuous, smooth, and differentiable everywhere to prevent the loss function from becoming fixed through mathematical operations. However, I believe we should consider this issue from the objective laws of biology: ~~our brain continuously replaces neurons (metabolism).~~ If we apply this mechanism to neural networks, rather than relying solely on mathematical approaches, could we solve the problem of the loss function becoming fixed?

> Notify:  
> The number of neurons in the brain tends to stabilize after adulthood, so the statement that 'our brain continuously replaces neurons (metabolism)' is a wrong description.  
> Thanks my friend 福酱<18963870758@163.com>.

## Existing Theories

[Spiking neural network(SNN)](<https://en.wikipedia.org/wiki/Spiking_neural_network>)  
[Papers with Code](<https://paperswithcode.com/task/architecture-search>)  
[ar5iv](<https://ar5iv.org/pdf/2304.10749.pdf>)

At present, it is necessary to carry out research on these theories, and first simulate their results based on the existing theories.

### Summarize

To date, we need to simulate the dynamic effects seen in biology, incorporating these effects into artificial neural networks. (Although in adulthood neurons no longer increase significantly in number), it is important to note that the strongest learning capabilities occur during youth and adolescence.

List of Challenges (Examples all taken from `src/model/dynamicNN.py`):

- [ ] How to make "dendrites" (`weights` in `dynamicNN.py`) automatically increase regularly?
  - [ ] 1. How to increase scientifically, because once the dendrites increase, it proves that the number of connected neurons has increased, how to perform this increase?
  - [ ] 2. The increase in dendrites proves that the passed down value will also increase, then `x` will also increase. How to ensure that x can increase or decrease accordingly?
  - [ ] 3. How to ensure that the passed values ​​`x` and `weights` are corresponding?
- [ ] How to make "neurons" (`CustomNeuron()` in `dynamicNN.py`) increase regularly?
  - [ ] 1. How to perform mitosis?
  - [ ] 2. How to construct the energy consumption mechanism of mitosis using Pytorch?
  - [ ] 3. How to connect the newly added neurons? This is the most difficult. If you add a new neuron to represent the next layer or the layer linked to it, you need to add the corresponding "dendrite" to obtain the data passed down by the neuron.
  - [ ] 4. How to ensure that the above (`weights`) will not be affected when dynamic changes occur.

## Cell division(focal point)

- [ ] Learn the biological theories related to cell division.

- [ ] Continue to write about Raphael’s 'principles'.

## How to use

This project uses Anaconda to manage the required packages,please make sure you have the relevant Anaconda environment.

### Step

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
