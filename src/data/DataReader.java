package ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork.src.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataReader {
    private final int rows = 28;
    private final int cols = 28;

    public List<Image> readData(String path) {
        List<Image> images = new ArrayList<>();

        try (BufferedReader dataReader = new BufferedReader(new FileReader(path))) {
            // Skip the header line
            String header = dataReader.readLine();
            System.out.println("Header skipped: " + header);

            String line;
            while ((line = dataReader.readLine()) != null) {
                String[] lineItems = line.split(",");
                
                // Debug: Print the first few values
                System.out.println("First 5 pixels: " + lineItems[0] + ", " + lineItems[1] + ", " + lineItems[2] + ", " + lineItems[3] + ", " + lineItems[4]);

                double[][] data = new double[rows][cols];
                
                int label = Integer.parseInt(lineItems[0].replace("\"", ""));

                // Fill the 28x28 pixel grid
                int i = 0; // Start from 0 (no label column)
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        data[row][col] = Double.parseDouble(lineItems[i]);
                        i++;
                    }
                }
                images.add(new Image(data, label));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println("Total images loaded: " + images.size());
        return images;
    }
}