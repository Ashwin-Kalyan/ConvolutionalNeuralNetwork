package ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.data.MatrixUtility;

public  class ConvolutionLayer extends Layer {
    private List<double[][]> _filters;
    private int _filterSize;
    private int _stepSize;
    private double _learningRate;
    
    private long SEED;

    private int _inLength;
    private int _inRows;
    private int _inCols;

    private List<double[][]> _lastInput;

    public ConvolutionLayer(List<double[][]> filters, int filterSize, int stepSize, int inLength, int inRows, int inCols, long seed, int numFilters, double learningRate) {
        this._filterSize = filterSize;
        this._stepSize = stepSize;
        this._learningRate = learningRate;

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
        _lastInput = list;
        List<double[][]> output = new ArrayList<>();
        
        for (int m = 0; m < list.size(); m++) {
            for (double[][] filter : _filters) {
                output.add(convolve(list.get(m), filter, _stepSize));
            }
        }

        return output;
    }

    public double[][] spaceArray(double[][] input) {
        if (_stepSize == 1) return input;

        int outRows = (input.length - 1) * _stepSize + 1;
        int outCols = (input[0].length - 1) * _stepSize + 1;

        double[][] output = new double[outRows][outCols];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                output[i * _stepSize][j * _stepSize] = input[i][j];
            }
        }

        return output;
    }

    public double[][] flipArrayHorizontal(double[][] array) {
        int rows = array.length;
        int cols = array[0].length;
        double[][] output = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[rows - i - 1][j] = array[i][j];
            }
        }

        return output;
    }

    public double[][] flipArrayVertical(double[][] array) {
        int rows = array.length;
        int cols = array[0].length;
        double[][] output = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[i][cols - j - 1] = array[i][j];
            }
        }

        return output;
    }

    private double[][] fullConvolve(double[][] input, double[][] filter) {
        int outputRows = (input.length + filter.length) + 1;
        int outputCols = (input[0].length + filter[0].length) + 1;
        
        int inRows = input.length;
        int inCols = input[0].length;
        
        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outputRows][outputCols];

        int outRow = 0;
        int outCol;

        for (int i = -fRows + 1; i < inRows; i++) {
            outCol = 0;

            for (int j = -fCols + 1; j < inCols; j++) {
                double sum = 0.0;

                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                        int inputRowIndex = i + x;
                        int inputColIndex = j + y;

                        if (inputRowIndex >= 0 && inputRowIndex < inRows && 
                            inputColIndex >= 0 && inputColIndex < inCols) {
                            double value = filter[x][y] * input[inputRowIndex][inputColIndex];
                            sum += value;  
                        }
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;
            }

            outRow++;
        }

        return output;
    }

    @Override
    public void backPropagation(double[] dLd0) {
        List<double[][]> matrixInput = vectorToMatrix(dLd0, _inLength, _inRows, _inCols);
        backPropagation(matrixInput);
    }

    @Override
    public void backPropagation(List<double[][]> dLd0) {
        List<double[][]> filtersDelta = new ArrayList<>();
        List<double[][]> dLdOPreviousLayer = new ArrayList<>();
        
        for (int f = 0; f < _filters.size(); f++) {
            filtersDelta.add(new double[_filterSize][_filterSize]);
        }

        for (int i = 0; i < _lastInput.size(); i++) {
            double[][] errorForInput = new double[_inRows][_inCols];

            for (int f = 0; f < _filters.size(); f++) {
                double[][] currFilter = _filters.get(f);
                double[][] error = dLd0.get(i * _filters.size() + f);
                double[][] spacedError = spaceArray(error);
                double[][] dLdF = convolve(_lastInput.get(i), spacedError, 1);

                double[][] delta = MatrixUtility.multiply(dLdF, _learningRate * -1);
                double[][] newTotalDelta = MatrixUtility.add(filtersDelta.get(f), delta);
                filtersDelta.set(f, newTotalDelta);

                double[][] flippedError = flipArrayHorizontal(flipArrayVertical(spacedError));
                errorForInput = MatrixUtility.add(errorForInput, fullConvolve(currFilter, flippedError));
            }

            dLdOPreviousLayer.add(errorForInput);
        }

        for (int f = 0; f < _filters.size(); f++) {
            double[][] modified = MatrixUtility.add(filtersDelta.get(f), _filters.get(f));
            _filters.set(f, modified);
        }

        if (_previousLayer != null) _previousLayer.backPropagation(dLdOPreviousLayer);
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
