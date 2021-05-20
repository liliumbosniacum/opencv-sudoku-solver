package com.lilium.sudoku.util;

import com.lilium.sudoku.mnist.evaluation.EvalUtil;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public final class Utils {
    private Utils() {}

    // region Implementation
    /**
     * Process forwarded image. Following is done:
     *      - Switch from RGB to GRAY
     *      - Invert every bit of an image (e.g. black to white and white to black)
     *
     * @param image Image to process.
     * @return Returns processed image.
     */
    public static Mat preProcessImage(final Mat image) {
        final Mat processed = new Mat(image.height(), image.width(), CvType.CV_8UC1);

        // RGB to GRAY
        Imgproc.cvtColor(image, processed, Imgproc.COLOR_RGB2GRAY);
        saveImage(processed, "processed-1-1.jpg");

        // Invert
        Core.bitwise_not(processed, processed);
        saveImage(processed, "processed-1-2.jpg");

        return processed;
    }

    /**
     * Used to mark outer rectangle and its corners.
     *
     * @param processedImage Image used for calculation of contours and corners.
     * @param originalImage Image on which marking is done.
     */
    public static void markOuterRectangleAndCorners(final Mat processedImage,
                                                    final Mat originalImage) {
        // Find contours of an image
        final List<MatOfPoint> allContours = new ArrayList<>();
        Imgproc.findContours(
                processedImage,
                allContours,
                new Mat(processedImage.height(), processedImage.width(), processedImage.type()),
                Imgproc.RETR_EXTERNAL, // We are looking for external contours
                Imgproc.CHAIN_APPROX_SIMPLE
        );

        // Find index of biggest contour
        final int biggestContourIndex = Utils.getBiggestPolygonIndex(allContours);

        // Mark outer contour (biggest one)
        markPolyCurve(allContours, biggestContourIndex, originalImage);

        // Find corner points and mark them
        final Point[] points = Utils.getPoints(allContours.get(biggestContourIndex));
        for (final Point point : points) {
            Imgproc.drawMarker(originalImage, point, new Scalar(255, 0, 0), 0, 30, 2);
        }
    }

    /**
     * Used to remove lines from processed image (lines forming the cells which hold digits).
     *
     * @param processedImage Processed image which should be black and white at this point.
     */
    public static void removeLines(final Mat processedImage) {
        final Mat lines = new Mat();

        // Detect lines
        Imgproc.HoughLinesP(
                processedImage,
                lines,
                1,
                Math.PI / 180,
                150,
                300,
                20
        );

        // Remove found lines. Removing in our case means just drawing over them with black color (our background is
        // also black).
        for (int r = 0; r < lines.rows(); r++) {
            double[] l = lines.get(r, 0);
            Imgproc.line(
                    processedImage,
                    new Point(l[0], l[1]),
                    new Point(l[2], l[3]),
                    new Scalar(0, 0, 255),
                    2,
                    Imgproc.FILLED,
                    0
            );
        }

        lines.release();
    }

    /**
     * Iterates over processed image and finds cells with values inside. With found cells it will try to estimate which
     * digit it contains. Found digits are stored in a matrix and returned.
     *
     * @param processedImage Processed image.
     * @param trainedNetwork Network that is capable of distinguishing between different digits.
     * @return Returns sudoku matrix.
     */
    public static int[][] getSudokuMatrix(final Mat processedImage, final MultiLayerNetwork trainedNetwork) {
        final int[][] matrix = new int[9][9];

        final int cellWidth = processedImage.width() / 9;
        final int cellHeight = processedImage.height() / 9;
        final Size cellSize = new Size(cellWidth, cellHeight);

        for (int row = 0; row < 9; row++) {
            for (int col = 0; col < 9; col++) {
                final double tempXPosition = col * cellWidth;
                final double tempYPosition = row * cellHeight;

                final Mat cell = new Mat(
                        processedImage,
                        new Rect(new Point(tempXPosition, tempYPosition), cellSize) // Which part to cut out
                ).clone();

                // Find non zero pixel count
                final int count = Core.countNonZero(cell);

                // If count is less then 150 we can assume that it is a blank cell (has no digit in it)
                if (count <= 150) {
                    matrix[row][col] = 0;
                } else { // We assume that there is a digit in the cell
                    // Save cell image for debugging
                    saveImage(cell, "digits\\" + System.nanoTime() +".jpg");

                    try {
                        // Estimate cell value
                        final int estimatedValue = trainedNetwork != null
                                ? EvalUtil.evaluateImage(cell, trainedNetwork)
                                : 1;
                        matrix[row][col] = estimatedValue;
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        }

        return matrix;
    }

    public static void printSolutionToImage(final Mat oldProcessedImage, final Mat debuggingImage, final int[][] solvedMatrix) {
        // Pre process image to be able to find contours
        final Mat processedImage = Utils.preProcessImage(debuggingImage);

        // Find contours of an image
        final List<MatOfPoint> allContours = new ArrayList<>();
        Imgproc.findContours(
                processedImage,
                allContours,
                new Mat(processedImage.height(), processedImage.width(), processedImage.type()),
                Imgproc.RETR_EXTERNAL, // We are looking for external contours
                Imgproc.CHAIN_APPROX_SIMPLE
        );

        // Find index of biggest contour
        final int biggestContourIndex = Utils.getBiggestPolygonIndex(allContours);

        // Find corner points and mark them
        final Point[] points = Utils.getPoints(allContours.get(biggestContourIndex));

        final int cellWidth = oldProcessedImage.width() / 9;
        final int cellHeight = oldProcessedImage.height() / 9;
        final Size cellSize = new Size(cellWidth, cellHeight);

        // Offset to apply on uncut image
        final double offsetX = points[1].x - oldProcessedImage.width();
        final double offsetY = points[2].y - oldProcessedImage.height();

        for (int row = 0; row < 9; row++) {
            for (int col = 0; col < 9; col++) {
                final double tempXPosition = (col * cellWidth) + offsetX;
                final double tempYPosition = (row * cellHeight) + offsetY;

                final Mat cell = new Mat(
                        oldProcessedImage,
                        new Rect(new Point((col * cellWidth), row * cellHeight), cellSize) // Which part to cut out
                ).clone();

                // Find non zero pixel count
                final int count = Core.countNonZero(cell);

                // If count is less then 150 we can assume that it is a blank cell (has no digit in it)
                if (count <= 150) {
                    Imgproc.putText (
                            debuggingImage,
                            String.valueOf(solvedMatrix[row][col]),
                            new Point(tempXPosition + (cell.width() / 3), tempYPosition + cell.height() - 10),
                            1,
                            8,
                            new Scalar(0, 0, 255),
                            6
                    );
                }
            }
        }
    }

    /**
     * Save image on provided path.
     *
     * @param imageToSave Image to be saved.
     * @param path Path on which image is saved.
     */
    public static void saveImage(final Mat imageToSave, final String path) {
        Imgcodecs.imwrite("D:\\Development\\IntelliJ\\opencv-sudoku-solver\\src\\main\\resources\\" + path, imageToSave);
    }

    /**
     * Load image from provided path.
     *
     * @param imagePath Image path.
     * @return Loads image from provided path.
     */
    public static Mat loadImage(final String imagePath) {
        return Imgcodecs.imread("D:\\Development\\IntelliJ\\opencv-sudoku-solver\\src\\main\\resources\\" + imagePath);
    }

    /**
     * Used to print forwarded matrix.
     *
     * @param matrix Matrix to print.
     */
    public static void printOutMatrix(final int[][] matrix) {
        System.out.println("######################################");
        for (int[] values : matrix) {
            System.out.print(" - ");
            for (int value : values) {
                System.out.print(value + " - ");
            }
            System.out.println("");
        }
        System.out.println("######################################");
    }
    // endregion

    // region Helpers
    /**
     * Used to get corner points of provided polygonal curve.
     *
     * @param poly Polygonal curve for which corner points are found.
     * @return Returns an array of found corner points.
     */
    private static Point[] getPoints(final MatOfPoint poly) {
        MatOfPoint2f approxPolygon = Utils.approxPolygon(poly);
        Point[] sortedPoints = new Point[4];

        if (!approxPolygon.size().equals(new Size(1, 4))) {
            return sortedPoints;
        }

        // Calculate the center of mass of our contour image using moments
        final Moments moment = Imgproc.moments(approxPolygon);
        final int centerX = (int) (moment.get_m10() / moment.get_m00());
        final int centerY = (int) (moment.get_m01() / moment.get_m00());

        // We need to sort corner points as there is not guarantee that we will always get them in same order
        for(int i=0; i<approxPolygon.rows(); i++){
            final double[] data = approxPolygon.get(i, 0);
            final double dataX = data[0];
            final double dataY = data[1];

            // Sorting is done in reverence to center points (centerX, centerY)
            if(dataX < centerX && dataY < centerY) {
                sortedPoints[0] = new Point(dataX, dataY);
            } else if(dataX > centerX && dataY < centerY) {
                sortedPoints[1] = new Point(dataX, dataY);
            } else if (dataX < centerX && dataY > centerY) {
                sortedPoints[2] = new Point(dataX, dataY);
            } else if (dataX > centerX && dataY > centerY) {
                sortedPoints[3] = new Point(dataX, dataY);
            }
        }

        return sortedPoints;
    }

    /**
     * Approximates a polygonal curve.
     *
     * @param poly Polygonal curve.
     * @return .
     */
    private static MatOfPoint2f approxPolygon(final MatOfPoint poly) {
        final MatOfPoint2f destination = new MatOfPoint2f();
        final MatOfPoint2f source = new MatOfPoint2f();
        poly.convertTo(source, CvType.CV_32FC2);

        // Approximates a polygonal curve with the specified precision
        Imgproc.approxPolyDP(
                source,
                destination,
                0.02 * Imgproc.arcLength(source, true),
                true
        );

        return destination;
    }

    /**
     * Used to find index of biggest polygonal curve.
     *
     * @param contours Contours for which index of biggest polygonal curve is calculated.
     * @return Returns an integer representing index of biggest polygonal curve.
     */
    private static int getBiggestPolygonIndex(final List<MatOfPoint> contours) {
        double maxValue = 0;
        var maxValueIndex = 0;
        for (var i = 0; i < contours.size(); i++) {
            final double contourArea = Imgproc.contourArea(contours.get(i));
            // If current value (contourArea) is bigger then maxValue then it becomes maxValue
            if (maxValue < contourArea) {
                maxValue = contourArea;
                maxValueIndex = i;
            }
        }

        return maxValueIndex;
    }

    /**
     * Mark polygonal curve with green colour.
     *
     * @param contours All contours.
     * @param index Index of biggest contour, if it is negative all contours will be drawn.
     * @param image Image on which contours are drawn.
     */
    private static void markPolyCurve(final List<MatOfPoint> contours,
                                      final int index,
                                      final Mat image) {
        Imgproc.drawContours(
                image,
                contours,
                index,
                new Scalar(124, 252, 0), // Green color
                3
        );
    }
    // endregion
}
