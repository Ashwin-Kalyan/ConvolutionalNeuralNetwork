package ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.layers;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer {
    private int _inLength;
    private int _outLength;

    private double[][] _weights;
    private long SEED;
    private final double leak = 0.01; 
    private double _learningRate;
    
    private double[] lastZ;
    private double[] lastX;

    public FullyConnectedLayer(int _inLength, int _outLength, long SEED, double _learningRate) {
        this._inLength = _inLength;
        this._outLength = _outLength;
        this.SEED = SEED;
        this._learningRate = _learningRate;

        _weights = new double[_inLength][_outLength];
        setRandomWeights();
    }

    public double[] fullyConnectedForwardPass(double[] input) {
        lastX = input;

        double[] z = new double[_outLength];
        double[] out = new double[_outLength];

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                z[j] += input[i] * _weights[i][j];
            }
        }

        lastZ = z;

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                out[j] = ReLU(z[j]);
            }
        }

        return out;
    }

    @Override
    public void backPropagation(double[] dLdO) {
        double[] dLdX = new double[_inLength];
        
        for (int k = 0; k < _inLength; k++) {
            double sum = 0;
            for (int j = 0; j < _outLength; j++) {
                double dOdz = derivativeReLU(lastZ[j]);
                sum += dLdO[j] * dOdz * _weights[k][j];
            }
            dLdX[k] = sum;
        }

        if (_previousLayer != null) {
            // Convert to matrix form expected by previous conv/pool layers
            List<double[][]> matrixGradients = vectorToMatrix(
                dLdX, 
                _previousLayer.getOutputLength(),
                _previousLayer.getOutputRows(),
                _previousLayer.getOutputCols()
            );
            _previousLayer.backPropagation(matrixGradients);
        }
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        double[] vector = matrixToVector(dLdO);
        backPropagation(vector);
    }
  
    public void setRandomWeights() {
        Random random = new Random(SEED);

        for (int i = 0; i< _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                _weights[i][j] = random.nextGaussian();
            }
        }
    }

    public double ReLU(double input) {
        return Math.max(0, input);
    }

    public double derivativeReLU(double input) {
        return input <= 0 ? leak : 1;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = fullyConnectedForwardPass(input);
        
        if (_nextLayer != null) return _nextLayer.getOutput(forwardPass);
        else return forwardPass; // If this is the last layer, return the output
    }

    @Override
    public int getOutputRows() {
        return 1; 
    }

    @Override
    public int getOutputCols() {
        return 1; 
    }

    @Override
    public int getOutputLength() {
        return _outLength;
    }

    @Override
    public int getInputElements() {
        return _outLength;
    }
}