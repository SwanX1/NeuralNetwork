# NNet

NNet is an unoptimized simple neural network library. It is meant for educational purposes, and is not meant for production (as it is unoptimized.)

## Installation

This library is meant for use with Bun, but if you copy the files `index.ts` and `validator.ts` into your project, it should work as normal both in NodeJS and Deno.
You may use this library for any purpose (see license below).

## Usage

### Creating a Neural Network

To create a neural network, you can use the static `create` method:

```typescript
import NeuralNetwork from 'nnet';

const options = {
  inputCount: 2,
  layerCount: 2,
  layerWidth: 3,
  outputCount: 1,
};

const neuralNetwork = NeuralNetwork.create(options);
```

You can customize the network architecture and activation function by providing appropriate options.

### Forward Propagation

To perform forward propagation and get the output of the neural network for a given input, use the `forward` method:

```typescript
const input = [0.5, 0.7];
const output = neuralNetwork.forward(input);
console.log(output);
```

### Training

To train the neural network using backpropagation, use the `train` method. It requires input data, expected output data, and an optional learning rate:

```typescript
const input = [0.1, 0.2];
const expectedOutput = [0.7];

neuralNetwork.train(input, expectedOutput, 0.1);
```

### Cloning

To create a copy of the neural network, use the `clone` method:

```typescript
const clonedNetwork = neuralNetwork.clone();
```

This creates a new instance with the same architecture and weights.

### Mutation

To introduce random mutations to the neural network's weights and biases, you can use the mutate method. This method allows you to specify the mutation rate, mutation range, and mutation factor:

```typescript

// Define the mutation rate (e.g., 0.1), mutation range (e.g., 3), and mutation factor (e.g., 0.5)
const mutationRate = 0.1;
const mutationRange = 3; // optional
const mutationFactor = 0.5; // optional

// Apply mutations to the neural network
neuralNetwork.mutate(mutationRate, mutationRange, mutationFactor);
```

 - Mutation Rate: The mutationRate parameter specifies the probability of mutation for each weight or bias. For example, a mutationRate of 0.1 means that, on average, 10% of the network's weights and biases will be mutated.
 - Mutation Range: The mutationRange parameter determines the range within which mutations can occur. It represents the maximum absolute value by which weights and biases can be adjusted during mutation.
 - Mutation Factor: The mutationFactor parameter scales the magnitude of mutations. Smaller values (e.g., 0.5) lead to smaller mutations, while larger values (e.g., 2) result in larger mutations.

This mutation process is particularly useful in evolutionary algorithms to introduce genetic diversity and explore different neural network architectures and configurations.

### JSON Serialization

The `NeuralNetwork` class supports JSON serialization and deserialization:

```typescript
const json = neuralNetwork.toJSON();
const loadedNetwork = NeuralNetwork.fromJSON(json);
```

This allows you to save and restore the network's architecture and weights.

## License

This library is released under CC-0, which basically means do whatever the fuck you want.