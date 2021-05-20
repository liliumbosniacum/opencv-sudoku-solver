package com.lilium.sudoku;

import com.lilium.sudoku.mnist.evaluation.EvalUtil;
import com.lilium.sudoku.util.SudokuUtil;
import com.lilium.sudoku.util.Utils;
import nu.pattern.OpenCV;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.opencv.core.Mat;

public class SudokuSolver {
    private static final String IMAGE = "sudoku.jpg";

    public static void main(final String args[]) {
        OpenCV.loadShared();

        // Load trained network
        final MultiLayerNetwork trainedNetwork = EvalUtil.loadModel();

        // Load and process image
        final Mat processedImage = Utils.preProcessImage(Utils.loadImage(IMAGE));
        // Load debugging image (same as one that is being processed)
        final Mat debuggingImage = Utils.loadImage(IMAGE);

        // Mark outer rectangle and corners (we do this just for debugging, and since it is cool :P)
        Utils.markOuterRectangleAndCorners(processedImage, debuggingImage);

        // Remove all lines from processed image
        Utils.removeLines(processedImage);

        // Get sudoku matrix with estimated values
        final int[][] sudokuMatrix = Utils.getSudokuMatrix(processedImage, trainedNetwork);

        // Solve and print out solution
        if (SudokuUtil.solve(sudokuMatrix)) {
            // Print solved matrix to the console
            Utils.printOutMatrix(sudokuMatrix);

            // Print solved matrix to the image
            Utils.printSolutionToImage(processedImage, debuggingImage, sudokuMatrix);
        } else {
            System.out.println("#### NOT ABLE TO SOLVE ####");
        }

        // Save processed and debugging images
        Utils.saveImage(processedImage, "processed-2.jpg");
        Utils.saveImage(debuggingImage, "debugging.jpg");
    }
}
