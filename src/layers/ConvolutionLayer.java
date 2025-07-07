package ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public  class ConvolutionLayer extends Layer {
    private List<double[][]> _filters;
    private int _filterSize;
    private int _stepSize;
    
    private long SEED;

    private int _inLength;
    private int _inRows;
    private int _inCols;

    public ConvolutionLayer(List<double[][]> filters, int filterSize, int stepSize, int inLength, int inRows, int inCols, long seed, int numFilters) {
        this._filterSize = filterSize;
        this._stepSize = stepSize;

        this.SEED = seed;

        this._inLength = inLength;
        this._inRows = inRows;
        this._inCols = inCols;

        generateRandomFilters(numFilters);
    }

    private void generateRandomFilters(int numFilters) {
        List<double[][]> filters = new ArrayList<>();
        Random random = new Random(SEED);

        for(int n = 0; n < numFilters; n++) {
            double[][] newFilter = new double[_filterSize][_filterSize];
            
            for (int i = 0; i < _filterSize; i++) {
                for (int j = 0; j < _filterSize; j++) {
                    double value = random.nextGaussian(); // Random value between 0 and 1
                    newFilter[i][j] = value;
                }
            }

            filters.add(newFilter);
        }

        _filters = filters;
    }

    private double[][] convolve(double[][] input, double[][] filter, int stepSize) {
        int outputRows = (input.length - filter.length) / stepSize + 1;
        int outputCols = (input[0].length - filter[0].length) / stepSize + 1;
        
        int inRows = input.length;
        int inCols = input[0].length;
        
        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outputRows][outputCols];

        int outRow = 0;
        int outCol;

        for (int i = 0; i <= inRows - fRows; i += stepSize) {
            outCol = 0;

            for (int j = 0; j <= inCols - fCols; j += stepSize) {
                double sum = 0.0;

                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                        int inputRowIndex = i + x;
                        int inputColIndex = j + y;

                        double value = filter[x][y] * input[inputRowIndex][inputColIndex];
                        sum += value;
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;
            }

            outRow++;
        }

        return output;
    }

    public List<double[][]> convolutionForwardPass(List<double[][]> list) {
        List<double[][]> output = new ArrayList<>();
        for (int m = 0; m < list.size(); m++) {
            for (double[][] filter : _filters) {
                output.add(convolve(list.get(m), filter, _stepSize));
            }
        }

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outputs = convolutionForwardPass(input);
        return _nextLayer.getOutput(outputs);
    }

    

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixInput = vectorToMatrix(input, _inLength, _inRows, _inCols);
        return getOutput(matrixInput);   
    }

    @Override
    public void backPropagation(double[] dLd0) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'backPropagation'");
    }

    @Override
    public void backPropagation(List<double[][]> dLd0) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'backPropagation'");
    }

    @Override
    public int getOutputLength() {
        return _filters.size() * _inLength;
    }

    @Override
    public int getOutputRows() {
        return (_inRows - _filterSize) / _stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (_inCols - _filterSize) / _stepSize + 1;
    }

    @Override
    public int getInputElements() {
        return getOutputCols() * getOutputRows() * getOutputLength();
    }
    
}
