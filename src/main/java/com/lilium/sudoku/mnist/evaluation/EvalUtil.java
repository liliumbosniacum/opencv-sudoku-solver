package com.lilium.sudoku.mnist.evaluation;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.opencv.core.Mat;

import java.io.IOException;

public final class EvalUtil {
    private static final int HEIGHT = 60;
    private static final int WIDTH = 60;
    private static final int N_OUTCOMES = 10;

    private static final NativeImageLoader IMAGE_LOADER = new NativeImageLoader(HEIGHT, WIDTH);
    private static final ImagePreProcessingScaler IMAGE_PRE_PROCESSING_SCALER = new ImagePreProcessingScaler(0, 1);

    private EvalUtil() {}

    public static int evaluateImage(final Mat image, final MultiLayerNetwork model) throws IOException {
        try (final INDArray input = Nd4j.create(1, HEIGHT * WIDTH)) {
            final INDArray imageArray = IMAGE_LOADER.asRowVector(image);
            IMAGE_PRE_PROCESSING_SCALER.transform(imageArray);
            input.putRow(0, imageArray);

            // Joining input and output matrices into a dataset
            final DataSet dataSet = new DataSet(input, Nd4j.create(1, N_OUTCOMES));

            // Data set contains one value only
            final INDArray predicted = model.output(
                    dataSet.get(0).getFeatures(),
                    false
            );
            final INDArray predictedValue = BooleanIndexing.firstIndex(
                    predicted,
                    Conditions.equals(predicted.maxNumber())
            );

            return Integer.parseInt(predictedValue.toString());
        }
    }

    public static MultiLayerNetwork loadModel() {
        try {
            return ModelSerializer.restoreMultiLayerNetwork(
                    "D:\\Development\\IntelliJ\\opencv-sudoku-solver\\src\\main\\resources\\models\\trained.tar"
            );
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
