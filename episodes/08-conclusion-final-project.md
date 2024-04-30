---
title: 'Wrap-up & Final Project'
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
