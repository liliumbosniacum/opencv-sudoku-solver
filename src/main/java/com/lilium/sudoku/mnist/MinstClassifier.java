package com.lilium.sudoku.mnist;

import org.apache.log4j.BasicConfigurator;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class MinstClassifier {
    private static final String RESOURCES_FOLDER_PATH = "D:\\Development\\IntelliJ\\opencv-sudoku-solver\\src\\main\\resources\\mnist";
    private static final int HEIGHT = 60;
    private static final int WIDTH = 60;
    private static final int N_SAMPLES_TRAINING = 204;
    private static final int N_SAMPLES_TESTING = 204;
    private static final int N_OUTCOMES = 10;

    public static void main(final String args[]) throws IOException {
        BasicConfigurator.configure();

        DataSetIterator dataSetIterator = getDataSetIterator(RESOURCES_FOLDER_PATH + "\\training", N_SAMPLES_TRAINING);
        buildModel(dataSetIterator);
    }

    private static DataSetIterator getDataSetIterator(final String folderPath, final int nSamples) throws IOException {
        final File folder = new File(folderPath);
        final File[] digitFolders = folder.listFiles();

        final NativeImageLoader nil = new NativeImageLoader(HEIGHT, WIDTH);
        final ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0,1);

        INDArray input = Nd4j.create(nSamples, HEIGHT*WIDTH);
        INDArray output = Nd4j.create(nSamples, N_OUTCOMES);

        int n = 0;
        for (final File digitFolder: digitFolders) {
            int labelDigit = Integer.parseInt(digitFolder.getName());
            final File[] imageFiles = digitFolder.listFiles();

            for (final File imgFile : imageFiles) {
                final INDArray img = nil.asRowVector(imgFile);
                scaler.transform(img);
                input.putRow(n, img);
                output.put(n, labelDigit, 1.0);
                n++;
            }
        }

        // Joining input and output matrices into a dataset
        final DataSet dataSet = new DataSet(input, output);
        // Convert the dataset into a list
        final List<DataSet> listDataSet = dataSet.asList();
        // Shuffle content of list randomly
        Collections.shuffle(listDataSet, new Random(System.currentTimeMillis()));

        // Build and return a dataset iterator
        return new ListDataSetIterator<>(listDataSet, 10);
    }

    private static void buildModel(DataSetIterator dsi) throws IOException {
        final MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Nesterovs(0.006, 0.9))
                .l2(1e-4).list()
                .layer(new DenseLayer.Builder()
                        .nIn(HEIGHT*WIDTH).nOut(1000).activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(1000).nOut(N_OUTCOMES).activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER).build())
                .build();

        final MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(500));

        model.fit(dsi);

        //Evaluation
        DataSetIterator testDsi = getDataSetIterator(RESOURCES_FOLDER_PATH+"\\testing", N_SAMPLES_TESTING);
        Evaluation eval = model.evaluate(testDsi);
        System.out.print(eval.stats());

        ModelSerializer.writeModel(model, "D:\\Development\\IntelliJ\\opencv-sudoku-solver\\src\\main\\resources\\models\\trained.tar", true);
    }
}
