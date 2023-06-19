/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.convolution;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Convolutional Neural Network implementation for crop type classification
 */
public class CropClassifierV2Tester {

    private static String MODEL_FOLD ="K3";
    private static String MODEL_NAME = "01132022232030_RELU_MSE_E14_model";

    private static  String[] CLASS_LABELS = new String[]{
        "maize", "other", "wheattypecrop"
    };

    static final String[] EXTENSIONS = new String[]{
        "gif", "png", "bmp", "jpg"// and other formats you need
    };
    private static final int IMAGE_HEIGHT = 56;
    private static final int IMAGE_WIDTH = 100;
    private static final int IMAGE_CHANNELS = 3;
    private static int OUTPUT_NUM = 3; // 2 croptype classification
    private static int BATCH_SIZE = 100; // number of samples that will be propagated through the network in each iteration

    private static MultiLayerNetwork net;

    private static final Logger LOGGER = LoggerFactory.getLogger(CropClassifierV2Tester.class);
    private static final String SURFACE_TERMINAL_BASE_PATH = "E:/02-17-20 Backup/Documents/IIASA";
    private static final String IIASA_TERMINAL_BASE_PATH = "H:/MyDocuments";
    private static final String TITAN_TERMINAL_BASE_PATH = "C:/Users/HP WorkStation/Documents/Documentos Titan/IIASA";

    /*TERMINAL_BASE_PATH stores the root path for input image database directory and output models directory*/
    private static final String TERMINAL_BASE_PATH = TITAN_TERMINAL_BASE_PATH;

    /*BASE_PATH stores the path for input Image Dataset directory*/
    private static final String BASE_PATH = TERMINAL_BASE_PATH + "/earthchallengeImages";

    private static final String EXPERIMENT_LABEL = "mwo";

    private static final String IMAGE_ROOT_PATH = BASE_PATH + "/crop_type_v2_data/" + EXPERIMENT_LABEL;


    private static final String MODEL_OUTPUT_PATH = BASE_PATH + "/models";

    // filter to identify images based on their extensions
    static final FilenameFilter IMAGE_FILTER = new FilenameFilter() {

        @Override
        public boolean accept(final File dir, final String name) {

            if (dir.isDirectory() && !(name.endsWith(".ini")) && !(name.endsWith(".db"))) {
                return true;
            }

            for (final String ext : EXTENSIONS) {

                if (name.endsWith("." + ext)) {
                    return (true);
                }
            }
            return (false);
        }
    };
    private static DataNormalization trainImageScaler;
    private static NativeImageLoader loader;

    public static void main(String[] args){

        System.out.println("ALLFINE1");
        loader = new NativeImageLoader(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS);
        trainImageScaler = initTrainImageScaler();
        try {
            net = initModel();
        } catch (IOException e) {
            e.printStackTrace();
        }

        testAllImages();
    }

    private static MultiLayerNetwork initModel() throws IOException {

        String modelPath = MODEL_OUTPUT_PATH + "/mwo/" + MODEL_FOLD + "/" + MODEL_NAME + ".zip";

        File model = new File(modelPath);

        if (!model.exists()) {
            throw new IOException("Can't find the model");
        } else {
            LOGGER.info("model file exists");
        }

        MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(model);
        return net;
    }


    private static void testAllImages() {

        String fileContents = "";
        String imagesPath = IMAGE_ROOT_PATH + "\\train";
        File dir = new File(imagesPath);
        int currentPart = 0;

        //LOGGER.info("imagesInPath = " +  dir.listFiles(IMAGE_FILTER).length);
        int truePositives = 0;
        int testSamples = 0;

        ArrayList<String> trueNegatives = new ArrayList<>();

        for (final File f : dir.listFiles(IMAGE_FILTER)) {

            LOGGER.info("class: " + f.getName());

            int no_samples =  f.listFiles(IMAGE_FILTER).length;
            int totalParts = 5;
            int samplesPerPart = (int) Math.floor(no_samples/totalParts);
            int init = currentPart * samplesPerPart;
            int end =  init + samplesPerPart;

            if(currentPart == (totalParts - 1)){
                end = no_samples;
            }

            for (int i = 0 ; i < no_samples ; i++) {

                File imgFile = f.listFiles(IMAGE_FILTER)[i];
                BufferedImage img = null;

                try {

                    img = ImageIO.read(imgFile);

                    if (img != null) {

                        // you probably want something more involved here
                        // to display in your U

                        String[] output = testImage(imgFile.getPath());
                        String ouputClassLabel = output[0];

                        testSamples += 1;


                        String[] parts = imgFile.getName().split("_");
                        String imgName = parts[parts.length - 2] + "_" + parts[parts.length - 1];

                        String sampleOutput = imgName + ", " + ouputClassLabel + "\n";
                        LOGGER.info(testSamples + ", " + sampleOutput );

                        fileContents += sampleOutput;

                        if (ouputClassLabel.equals(f.getName())) {
                            truePositives++;
                        } else {

                            trueNegatives.add(imgFile.getName());
                        }

                    }

                } catch (final IOException e) {
                    // handle errors here
                }
            }

        }

        LOGGER.info("valSamples: " + testSamples);
        LOGGER.info("truePositives: " + truePositives);

        DecimalFormat df = new DecimalFormat("0.0000");
        LOGGER.info("Accuracy: " + df.format((float) truePositives / (float) testSamples));


        saveResultsFile(fileContents);

        LOGGER.info("");
    }

    private static void saveResultsFile(String fileContents) {
        String resultsFile = BASE_PATH  +"/models/" + EXPERIMENT_LABEL + "/" + MODEL_FOLD + "/" + MODEL_NAME+  "_TRAIN_RESULTS.txt";

        try {
            FileWriter fw = new FileWriter(resultsFile);
            BufferedWriter bw = new BufferedWriter(fw);
            bw.write(fileContents);
            bw.newLine();
            bw.flush();

        } catch (IOException e) {
            LOGGER.error("save results FileWriter" + e.toString());
        }

        LOGGER.info("RESULTS FILE  EXPORTED");

    }


    private static String[] testImage(String newImagePath) {

        String[] result = new String[2];
        String prediction = "", probability = "";

        try {

            INDArray image = loader.asMatrix(new File(newImagePath));

            trainImageScaler.transform(image);

            int[] outputPrediction = net.predict(image);
            prediction = CLASS_LABELS[outputPrediction[0]];

            INDArray output = net.output(image);
            probability = String.valueOf(output.getDouble(outputPrediction[0]) * 100);

        } catch (IOException e) {
            e.printStackTrace();
        }

        result[0] = prediction;
        result[1] = probability;
        LOGGER.info("");

        return result;
    }

    private static DataNormalization initTrainImageScaler() {

        String trainDatPath = IMAGE_ROOT_PATH + "\\train";
        File trainData = new File(trainDatPath);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // use parent directory name as the image label
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, new Random(1234));

        ImageRecordReader trainRR = new ImageRecordReader(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, labelMaker);

        try {
            trainRR.initialize(trainSplit);
        } catch (IOException e) {
            e.printStackTrace();
        }

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, BATCH_SIZE, 1, OUTPUT_NUM);

        // pixel values from 0-255 to 0-1 (min-max scaling)
        ImagePreProcessingScaler trainImageScaler = new ImagePreProcessingScaler();
        trainImageScaler.fit(trainIter);

        return trainImageScaler;
    }


}
