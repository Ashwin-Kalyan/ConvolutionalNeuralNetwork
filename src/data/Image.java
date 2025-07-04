package ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.data;

public class Image {
    private double[][] data;
    private int label;

    public Image(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    public double[][] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Label: ").append(label).append("\n");
        
        // Print the 28x28 grid in a square format
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                // Format each pixel to 3 characters wide for alignment
                sb.append(String.format("%3.0f ", data[i][j]));
            }
            sb.append("\n"); // New line after each row
        }
        return sb.toString();
    }
}