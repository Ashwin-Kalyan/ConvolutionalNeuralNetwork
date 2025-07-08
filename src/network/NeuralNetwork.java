package ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.network;

import java.util.ArrayList;
import java.util.List;

import ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.data.Image;
import ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.data.MatrixUtility;
import ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.layers.Layer;

public class NeuralNetwork {
    List<Layer> _layers;
    double scaleFactor;

    public NeuralNetwork(List<Layer> layers, double scaleFactor) {
        this._layers = layers;
        this.scaleFactor = scaleFactor;
        linkLayers();
    }

    private void linkLayers() {
        if (_layers.size() <= 1) return;

        for (int i = 0; i < _layers.size(); i++) {
            Layer currentLayer = _layers.get(i);
            
            // Set previous layer (except for first layer)
            if (i > 0) {
                currentLayer.setPreviousLayer(_layers.get(i - 1));
            }
            
            // Set next layer (except for last layer)
            if (i < _layers.size() - 1) {
                currentLayer.setNextLayer(_layers.get(i + 1));
            }
        }
    }

    public double[] getErrors(double[] networkOutput, int correctAnswer) {
        int numClasses = networkOutput.length;
        double[] expected = new double[numClasses];
        expected[correctAnswer] = 1;
        return MatrixUtility.add(networkOutput, MatrixUtility.multiply(expected, -1));
    }

    private int getMaxIndex(double[] in) {
        double max = 0;
        int index = 0;

        for (int i = 0; i < in.length; i++) {
            if (in[i] >= max) {
                max = in[i];
                index = i;
            }
        }
        return index;
    }

    public int guess(Image image) {
        List<double[][]> inList = new ArrayList<>();
        inList.add(MatrixUtility.multiply(image.getData(), (1.0 / scaleFactor))); // Normalize the input

        double[] out = _layers.get(0).getOutput(inList);
        int guess = getMaxIndex(out);
        return guess;
    }

    public float test(List<Image> images) {
        int correct = 0;

        for (Image image : images) {
            int guess = guess(image);
            if (guess == image.getLabel()) correct++;
        }

        return (((float) correct/images.size()) * 100) ;
    }

    public void train(List<Image> images) {
        for (Image image : images) {
            List<double[][]> inList = new ArrayList<>();
            inList.add(image.getData());

            double[] out = _layers.get(0).getOutput(inList);
            double[] dldO = getErrors(out, image.getLabel());

            _layers.get(_layers.size() - 1).backPropagation(dldO);
        }
    }
}