package ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src;

import java.util.List;
import static java.util.Collections.shuffle;
import ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.data.DataReader;
import ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.data.Image;
import ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.network.NetworkBuilder;
import ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.network.NeuralNetwork;

public class Main {
    public static void main(String[] args) {
        System.out.println("Starting Convolutional Neural Network...");
        List<Image> trainImages = new DataReader().readData("C:\\Users\\ashwi\\OneDrive\\Desktop\\GitHub\\ConvolutionalNeuralNetwork\\ConvolutionalNeuralNetwork\\src\\train.csv");
         List<Image> testImages = new DataReader().readData("C:\\Users\\ashwi\\OneDrive\\Desktop\\GitHub\\ConvolutionalNeuralNetwork\\ConvolutionalNeuralNetwork\\src\\test.csv");
        // for (int i = 0; i < 42000; i++){
        //     if (!images.isEmpty()) System.out.printf(images.get(i).toString());
        // }  
        NetworkBuilder builder = new NetworkBuilder(28, 28, 256*100);
        builder.addConvolutionLayer(8, 5, 1, 0.1, 123);
        builder.addMaxPoolLayer(3, 2);
        builder.addFullyConnectedLayer(10, 0.1, 123);
        NeuralNetwork net = builder.build();

        float rate = net.test(trainImages);
        System.out.printf("Pre-Training Test Accuracy: %.2f%%\n", rate);

        int epochs = 3;
        for (int i = 0; i < epochs; i++) {
            shuffle(trainImages);
            net.train(trainImages);
            rate = net.test(trainImages);
            System.out.printf("Epoch %d --- Training Accuracy: %.2f%%\n", i + 1, rate);
        }
    }
}
