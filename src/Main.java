package ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src;

import java.util.List;

import ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.data.DataReader;
import ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.data.Image;

public class Main {
    public static void main(String[] args) {
        List<Image> images = new DataReader().readData("C:\\Users\\ashwi\\OneDrive\\Desktop\\GitHub\\ConvolutionalNeuralNetwork\\ConvolutionalNeuralNetwork\\src\\train.csv");
        for (int i = 0; i < 42000; i++){
            if (!images.isEmpty()) System.out.printf(images.get(i).toString());
        }  
    }
}
