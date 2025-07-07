package ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.layers;

import java.util.ArrayList;
import java.util.List;

public abstract class Layer {
    protected Layer _nextLayer;
    protected Layer _previousLayer;

    public abstract double[] getOutput(List<double[][]> input);
    public abstract double[] getOutput(double[] input);  

    public abstract void backPropagation(double[] dLd0);
    public abstract void backPropagation(List<double[][]> dLd0); 

    public abstract int getOutputLength();
    public abstract int getOutputRows();
    public abstract int getOutputCols();
    public abstract int getInputElements();

    public Layer getNextLayer() {
        return _nextLayer;
    }

    public Layer getPreviousLayer() {
        return _previousLayer;
    }

    public void setNextLayer(Layer nextLayer) {
        this._nextLayer = nextLayer;
    }

    public void setPreviousLayer(Layer previousLayer) {
        this._previousLayer = previousLayer;
    }

    public double[] matrixToVector(List<double[][]> input) {
        int length = input.size();
        int rows = input.get(0).length;
        int cols = input.get(0)[0].length;

        double[] vector = new double[length * rows * cols];

        int i = 0;
        for (int j = 0; j < length; j++) {
            for (int k = 0; k < rows; k++) {
                for (int l = 0; l < cols; l++) {
                    vector[i] = input.get(j)[k][l];
                    i++;
                }
            }
        }

        return vector;
    }

    public List<double[][]> vectorToMatrix(double[] input, int length, int rows, int cols) {
        List<double[][]> out = new ArrayList<>();

        int i = 0;
        for (int j = 0; j < length; j++) {
            double[][] matrix  = new double[rows][cols];
            
            for (int k = 0; k < rows; k++) {
                for (int l = 0; l < cols; l++) {
                    matrix[k][l] = input[i];
                    i++;
                }
            }

            out.add(matrix);
        }

        return out;
    }
}