---
title: 'Domain Specific LLMs'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- How do you compile a convolutional neural network (CNN)?
- What is a loss function?
- What is an optimizer?
- How do you train (fit) a CNN?
- How do you evaluate a model during training?
- What is overfitting?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain the difference between compiling and training (fitting) a CNN.
- Know how to select a loss function for your model.
- Understand what an optimizer is.
- Define the terms: learning rate, batch size, epoch.
- Understand what loss and accuracy are and how to monitor them during training.
- Explain what overfitting is and what to do about it.

::::::::::::::::::::::::::::::::::::::::::::::::

### Step 5. Choose a loss function and optimizer and compile model

We have designed a convolutional neural network (CNN) that in theory we should be able to train to classify images. 

We now need to **compile** the model, or set up the rules and strategies for how the network will learn. To do this, we select an appropriate loss function and optimizer to use during training (fitting). 

Recall how we compiled our model in the introduction:
```
## compile the model
#model_intro.compile(optimizer = 'adam', 
#                    loss = keras.losses.CategoricalCrossentropy(), 
#                    metrics = ['accuracy'])
```              

#### Loss function

The **loss function** tells the training algorithm how wrong, or how 'far away' from the true value the predicted value is. The purpose of loss functions is to compute the quantity that a model should seek to minimize during training. Which class of loss functions you choose depends on your task. 

**Loss for classification**

For classification purposes, there are a number of probabilistic losses to choose from. We chose `CategoricalCrossentropy` because we want to compute the crossentropy loss between our one-hot encoded class labels and the model predictions. This loss function is appropriate to use when the data has two or more label classes.

The loss function is defined by the `tf.keras.losses.CategoricalCrossentropy` class.

More information about loss functions can be found in the Keras [loss documentation].


#### Optimizer

Somewhat coupled to the loss function is the **optimizer**. The optimizer here refers to the algorithm with which the model learns to optimize on the provided loss function.

We need to choose which optimizer to use and, if this optimizer has parameters, what values to use for those. Furthermore, we specify how many times to present the training samples to the optimizer. In other words, the optimizer is responsible for taking the output of the loss function and then applying some changes to the weights within the network. It is through this process that the “learning” (adjustment of the weights) is achieved.

```
## compile the model
#model_intro.compile(optimizer = 'adam', 
#                    loss = keras.losses.CategoricalCrossentropy(), 
#                    metrics = ['accuracy'])
``` 

**Adam** 

Here we picked one of the most common optimizers demonstrated to work well for most tasks, the **Adam** optimizer. Similar to other hyperparameters, the choice of optimizer depends on the problem you are trying to solve, your model architecture, and your data. Adam is a good starting point though, which is why we chose it. Adam has a number of parameters, but the default values work well for most problems so we will use it with its default parameters.
