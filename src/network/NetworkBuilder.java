package ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.network;

import java.util.ArrayList;
import java.util.List;

import ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.layers.ConvolutionLayer;
import ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.layers.FullyConnectedLayer;
import ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.layers.Layer;
import ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.layers.MaxPoolLayer;

public class NetworkBuilder {
    private NeuralNetwork net;
    private int _inputRows;
    private int _inputCols;
    List<Layer> _layers;
    private double _scaleFactor;

    public NetworkBuilder(int inputRows, int inputCols, double scaleFactor) {
        this._inputRows = inputRows;
        this._inputCols = inputCols;
        this._scaleFactor = scaleFactor;
        _layers = new ArrayList<>();
    }

    public void addConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate, long SEED) {
        if (_layers.isEmpty()) _layers.add(new ConvolutionLayer(filterSize, stepSize, 1, _inputRows, _inputCols, SEED, numFilters, learningRate));
        Layer prev = _layers.get(_layers.size() - 1);
        _layers.add(new ConvolutionLayer(filterSize, stepSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols(), SEED, numFilters, learningRate));
    }

    public void addMaxPoolLayer(int windowSize, int stepSize) {
        if (_layers.isEmpty()) _layers.add(new MaxPoolLayer(stepSize, windowSize, 1, _inputRows, _inputCols));
        Layer prev = _layers.get(_layers.size() - 1);
        _layers.add(new MaxPoolLayer(stepSize, windowSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols()));
    }

    public void addFullyConnectedLayer(int outLength, double learningRate, long SEED) {
        if (_layers.isEmpty()) {
            // For first layer, flatten 28x28 input
            _layers.add(new FullyConnectedLayer(_inputRows * _inputCols, outLength, SEED, learningRate));
        } else {
            Layer prev = _layers.get(_layers.size() - 1);
            // Flatten previous layer's output
            int flattenedSize = prev.getOutputLength() * prev.getOutputRows() * prev.getOutputCols();
            _layers.add(new FullyConnectedLayer(flattenedSize, outLength, SEED, learningRate));
        }
    }

    public NeuralNetwork build() {
        net = new NeuralNetwork(_layers, _scaleFactor);
        return net;
    }
}
