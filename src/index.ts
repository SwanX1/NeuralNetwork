import { assertObject, assertValue } from "./validator";

const activationFns: Record<string, (x: number) => number> = {
  "x => 1 / (1 + Math.exp(-x))": x => 1 / (1 + Math.exp(-x)),
  "x => Math.max(0, x)": x => Math.max(0, x),
  "x => x": x => x,
  "x => Math.tanh(x)": x => Math.tanh(x),
};

/**
 * A simple neural network with configurable architecture and activation function.
 * This implementation uses a sigmoid activation function by default, but can be configured to use other activation functions.
 * Use the {@link create} method to create a new neural network with empty neurons. The constructor will not hold your hand.
 * 
 * Methods are provided for forward propagation ({@link forward}), training ({@link train}), and mutation ({@link mutate}).
 * All of these mutate the neural network in place, if this isn't desired, use the {@link clone} method to create a copy of the neural network.
 * The {@link toJSON} and {@link fromJSON} methods can be used to serialize and deserialize the neural network.
 * 
 * Do not use this implementation for production purposes, it is intended for educational purposes only.
 */
export default class NeuralNetwork {
  private readonly inputCount: number;
  private readonly weights: number[][][];
  private readonly bias: number[][];
  private readonly activation: (x: number) => number;

  /**
   * Creates a new NeuralNetwork instance with the provided options.
   * @param options - Configuration options for the neural network.
   */
  constructor(options: {
    inputCount: number;
    weights: number[][][];
    bias: number[][];
    activation: (x: number) => number;
  }) {
    assertObject(options, {
      inputCount: "number",
      weights: "number[][][]",
      bias: "number[][]",
      activation: "function",
    });

    this.inputCount = options.inputCount;
    this.weights = options.weights;
    this.bias = options.bias;
    this.activation = options.activation;
  }

  /**
   * Creates a neural network with random weights and biases based on the given architecture.
   * @param options - Configuration options for creating the neural network.
   * @returns A new NeuralNetwork instance with random weights and biases.
   */
  public static create(options: {
    inputCount: number;
    layerCount: number;
    layerWidth: number;
    outputCount: number;
    activation?: (x: number) => number;
  }): NeuralNetwork {
    assertObject(options, {
      inputCount: "number",
      layerCount: "number",
      layerWidth: "number",
      outputCount: "number",
      activation: "function",
    });

    // Default to sigmoid activation function
    if (!options.activation) options.activation = x => 1 / (1 + Math.exp(-x));

    const weights: number[][][] = [];
    const bias: number[][] = [];

    // Initialize weights and biases for each layer and neuron
    for (let layer = 0; layer < options.layerCount; layer++) {
      weights.push([]);
      bias.push([]);
      for (let neuron = 0; neuron < options.layerWidth; neuron++) {
        weights[layer].push([]);
        bias[layer].push(0);
        for (let input = 0; input < (layer === 0 ? options.inputCount : options.layerWidth); input++) {
          weights[layer][neuron].push(0);
        }
      }
    }

    // Initialize weights and biases for the output layer
    for (let neuron = 0; neuron < options.outputCount; neuron++) {
      weights.push([]);
      bias.push([]);
      for (let input = 0; input < options.layerWidth; input++) {
        if (typeof weights[options.layerCount][neuron] === "undefined") weights[options.layerCount][neuron] = [];
        weights[options.layerCount][neuron].push(0);
        bias[options.layerCount][neuron] = 0;
      }
    }

    return new NeuralNetwork({ inputCount: options.inputCount, weights, bias, activation: options.activation });
  }

  /**
   * Performs a forward pass through the neural network.
   * @param input - The input values for the network.
   * @returns The output values produced by the network.
   * @throws Error if the input size does not match the neural network's input size.
   */
  public forward(input: number[]): number[] {
    assertValue(input, "number[]");

    if (input.length !== this.inputCount) {
      throw new Error("Input size does not match the neural network's input size.");
    }

    let output: number[] = [];

    for (let layer = 0; layer < this.weights.length; layer++) {
      for (let neuron = 0; neuron < this.weights[layer].length; neuron++) {
        let sum = 0;
        for (let inputIndex = 0; inputIndex < this.weights[layer][neuron].length; inputIndex++) {
          sum += this.weights[layer][neuron][inputIndex] * input[inputIndex];
        }
        sum += this.bias[layer][neuron];
        output.push(this.activation(sum));
      }
      input = output;
      output = [];
    }

    return input;
  }

  /**
   * Creates a clone of the current neural network.
   * @returns A new NeuralNetwork instance with the same architecture and weights.
   */
  public clone(): NeuralNetwork {
    const weights = this.weights.map(layer => layer.map(neuron => neuron.slice()));
    const bias = this.bias.map(layer => layer.slice());

    return new NeuralNetwork({ inputCount: this.inputCount, weights, bias, activation: this.activation });
  }

  /**
   * Mutates the neural network's weights and biases randomly based on the given rate.
   *
   * @param {number} rate - The mutation rate, indicating the probability of mutation for each weight or bias.
   * @param {number} [mutationRange=3] - The range within which weights and biases can be mutated.
   * @param {number} [mutationFactor=0.5] - The factor by which weights and biases are mutated.
   * @returns {NeuralNetwork} - The mutated NeuralNetwork instance.
   */
  public mutate(rate: number, mutationRange = 3, mutationFactor = 0.5): NeuralNetwork {
    assertValue(rate, "number");
    assertValue(mutationRange, "number");
    assertValue(mutationFactor, "number");

    for (const layerWeights of this.weights) {
      for (const neuronWeights of layerWeights) {
        for (let k = 0; k < neuronWeights.length; k++) {
          if (Math.random() < rate) neuronWeights[k] += (Math.random() - mutationFactor) * mutationRange;
        }
      }
    }

    for (const layerBias of this.bias) {
      for (let j = 0; j < layerBias.length; j++) {
        if (Math.random() < rate) layerBias[j] += (Math.random() - mutationFactor) * mutationRange;
      }
    }

    return this;
  }

  /**
   * Trains the neural network using input and expected output data.
   * @param input - The input data for training.
   * @param expectedOutput - The expected output data for training.
   * @param learningRate - The learning rate for adjusting weights and biases during training (default: 0.1).
   * @throws Error if the input size does not match the neural network's input size.
   */
  public train(input: number[], expectedOutput: number[], learningRate: number = 0.1) {
    assertValue(input, "number[]");
    assertValue(expectedOutput, "number[]");
    assertValue(learningRate, "number");

    if (input.length !== this.inputCount) {
      throw new Error(`Input size does not match the neural network's input size. Expected ${this.inputCount}, got ${input.length}.`);
    }

    const layerOutputs: number[][] = [];
    layerOutputs.push(input);

    // Forward pass: compute layer outputs
    for (let layer = 0; layer < this.weights.length; layer++) {
      const currentLayerOutput: number[] = [];
      for (let neuron = 0; neuron < this.weights[layer].length; neuron++) {
        let sum = this.bias[layer][neuron];
        for (let inputIndex = 0; inputIndex < this.weights[layer][neuron].length; inputIndex++) {
          sum += this.weights[layer][neuron][inputIndex] * layerOutputs[layer][inputIndex];
        }
        currentLayerOutput.push(this.activation(sum));
      }
      layerOutputs.push(currentLayerOutput);
    }

    // Backpropagation: compute errors and update weights and biases
    const errors: number[][] = [];
    for (let i = 0; i < this.weights.length; i++) {
      const layerErrors: number[] = new Array(this.weights[i].length).fill(0);
      errors.push(layerErrors);
    }

    for (let i = 0; i < expectedOutput.length; i++) {
      const output = layerOutputs[layerOutputs.length - 1][i];
      errors[errors.length - 1][i] = (expectedOutput[i] - output) * output * (1 - output);
    }

    for (let layer = this.weights.length - 2; layer >= 0; layer--) {
      for (let neuron = 0; neuron < this.weights[layer].length; neuron++) {
        let sum = 0;
        for (let k = 0; k < this.weights[layer + 1].length; k++) {
          sum += this.weights[layer + 1][k][neuron] * errors[layer + 1][k];
        }
        errors[layer][neuron] = layerOutputs[layer + 1][neuron] * (1 - layerOutputs[layer + 1][neuron]) * sum;
      }
    }

    for (let layer = 0; layer < this.weights.length; layer++) {
      for (let neuron = 0; neuron < this.weights[layer].length; neuron++) {
        for (let inputIndex = 0; inputIndex < this.weights[layer][neuron].length; inputIndex++) {
          this.weights[layer][neuron][inputIndex] += learningRate * errors[layer][neuron] * layerOutputs[layer][inputIndex];
        }
        this.bias[layer][neuron] += learningRate * errors[layer][neuron];
      }
    }
  }

  /**
   * Converts the neural network to a JSON string representation.
   * @returns A JSON string representing the neural network's architecture and weights.
   */
  public toJSON(): string {
    return JSON.stringify({
      inputCount: this.inputCount,
      activation: this.activation.toString(),
      weights: this.weights,
      bias: this.bias,
    });
  }

  /**
   * Creates a new NeuralNetwork instance from a JSON string representation.
   * @param json - A JSON string representing a neural network.
   * @returns A new NeuralNetwork instance reconstructed from the JSON data.
   * @throws Error if the JSON data is invalid or the activation function is not recognized.
   */
  public static fromJSON(json: string, activation?: (x: number) => number): NeuralNetwork {
    assertValue(json, "string");
    if (typeof activation !== "undefined") {
      assertValue(activation, "function");
    }

    let nn;
    try {
      nn = JSON.parse(json);
    } catch (e) {
      throw new Error("Invalid JSON data.", { cause: e });
    }
    
    try {
      assertObject(nn, {
        inputCount: "number",
        weights: "number[][][]",
        bias: "number[][]",
        activation: "string",
      });
    } catch (e) {
      throw new Error("Invalid neural network data.", { cause: e });
    }

    const activationFn = activationFns[nn.activation] ?? activation;
    if (!activationFn) throw new Error("Activation function is not recognized, please set a custom activation function.");

    return new NeuralNetwork({
      inputCount: nn.inputCount,
      weights: nn.weights,
      bias: nn.bias,
      activation: activationFn,
    });
  }
}
