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

import org.datavec.image.transform.*;
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
import org.json.JSONObject;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
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
public class CropClassifierV2 {


    private static int[] WIDTH_VALUES = {100, 150, 200, 250, 300, 350, 400, 450, 500};
    private static int[] HEIGHT_VALUES = {56, 84, 112, 140, 169, 197, 225, 253, 281}; //width and height of the picture in px
    private static int SIZE_INDEX = 0;

    /*Possible activation function options*/
    private static Activation[] ACTIVATION_FNS = {Activation.RELU, Activation.IDENTITY, Activation.TANH}; // Activation.SIGMOID CRASHED
    private static int ACITVATION_INDEX = 0;

    /*Possible loss function options for output layer*/
    private static LossFunctions.LossFunction[] LOSS_FNS = {LossFunctions.LossFunction.MSE, LossFunctions.LossFunction.POISSON, LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR};
    private static int LOSS_INDEX = 0;


    private static int CHANNELS = 3;   // single channel for grayscale images
    private static int OUTPUT_NUM = 3; // 2 croptype classification
    private static int BATCH_SIZE = 100; // number of samples that will be propagated through the network in each iteration
    private static int EPOCHS = 15;    // number of training epochs
    private static int K_FOLDS = 1; // number of times to train the experiment


    private static MultiLayerNetwork net;

    private static final Logger LOGGER = LoggerFactory.getLogger(CropClassifierV2.class);
    private static final String SURFACE_TERMINAL_BASE_PATH = "E:/02-17-20 Backup/Documents/IIASA";
    private static final String IIASA_TERMINAL_BASE_PATH = "E:\\titan_documents\\IIASA";
    /*TERMINAL_BASE_PATH stores the root path for input image database directory and output models directory*/
    private static final String TERMINAL_BASE_PATH = IIASA_TERMINAL_BASE_PATH;

    /*BASE_PATH stores the path for input Image Dataset directory*/
    private static final String BASE_PATH = TERMINAL_BASE_PATH + "/earthchallengeImages";

    private static final String EXPERIMENT_LABEL = "mwo";

    private static final String IMAGE_ROOT_PATH = BASE_PATH + "/crop_type_v4_data/" + EXPERIMENT_LABEL;


    private static final String MODEL_OUTPUT_PATH = BASE_PATH + "/models";

    private static final String DIR_TYPE = "DIR_TYPE";
    private static final String FILE_TYPE = "FILE_TYPE";



    private static ArrayList<String> getChildFiles(String parentPath, String type){

        ArrayList<String> classes = new ArrayList<String>();

        FilenameFilter filter;

        if (type == DIR_TYPE){
            filter = new FilenameFilter() {
                @Override
                public boolean accept(File current, String name) {
                    return new File(current, name).isDirectory();
                }
            };

        }else{
            filter = new FilenameFilter() {
                @Override
                public boolean accept(File current, String name) {
                    return new File(current, name).isFile();
                }
            };

        }

        File parentDir = new File(parentPath);
        String[] directories = parentDir.list(filter);

        for (String child: directories) {

            classes.add(child);

        }

        return classes;

    }


    public static void main(String[] args) throws Exception {

        ArrayList<String> subsetsLabels = getChildFiles(IMAGE_ROOT_PATH, DIR_TYPE);

        ArrayList<String> classLabels = getChildFiles(IMAGE_ROOT_PATH + "/" + subsetsLabels.get(0), DIR_TYPE);

        Random randNumGen = new Random(42);

//        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "cropclassifier_stats.dl4j"));

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
//
        String localModelDirectoryPath = (MODEL_OUTPUT_PATH + "\\" + EXPERIMENT_LABEL);
        LOGGER.info("localModelDirectoryPath : " + localModelDirectoryPath);
        checkCreateFile(localModelDirectoryPath);

        for (int k = 0; k < K_FOLDS; k++) {

            k = 1;

            /*Creates a directory for each iteration of the experiment*/
            String foldDirectoryPath = (localModelDirectoryPath + "\\2023\\K" + k);
            checkCreateFile(foldDirectoryPath);

            LOGGER.info("Data vectorization...");
            // vectorization of train data
            String trainPath = IMAGE_ROOT_PATH + "\\train";
            String testPath = IMAGE_ROOT_PATH + "\\test";
            File trainData = new File(trainPath);
            File testData = new File(testPath);

            /*Creates Train and Test Dataset iterator for further model training*/
            ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // use parent directory name as the image label

            ImageRecordReader trainRR = new ImageRecordReader(HEIGHT_VALUES[SIZE_INDEX], WIDTH_VALUES[SIZE_INDEX], CHANNELS, labelMaker);
            FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
            trainRR.initialize(trainSplit);

            DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, BATCH_SIZE, 1, OUTPUT_NUM);

            // pixel values from 0-255 to 0-1 (min-max scaling)
            DataNormalization imageScaler = new ImagePreProcessingScaler();
            imageScaler.fit(trainIter);

            trainIter.setPreProcessor(imageScaler);

            FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
            ImageRecordReader testRR = new ImageRecordReader(HEIGHT_VALUES[SIZE_INDEX], WIDTH_VALUES[SIZE_INDEX], CHANNELS, labelMaker);
            testRR.initialize(testSplit);

            DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, BATCH_SIZE, 1, OUTPUT_NUM);
            testIter.setPreProcessor(imageScaler); // same ImageRecordReader normalization for better results

            LOGGER.info("Network configuration and training...");
            // reduce the learning rate as the number of training epochs increases
            // iteration #, learning rate
            Map<Integer, Double> learningRateSchedule = new HashMap<>();
            learningRateSchedule.put(0, 0.06);
            learningRateSchedule.put(200, 0.05);
            learningRateSchedule.put(600, 0.028);
            learningRateSchedule.put(800, 0.0060);
            learningRateSchedule.put(1000, 0.001);


            /*MultiLayerConfiguration sets the CNN arquitecture configuration for training,
             * in this case, consists in 2 convolution layers, then a fully connected layer and finally an output layer*/

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
                .weightInit(WeightInit.XAVIER)
                .list()

                //CONV LAYER 1: Kernel window size {3,3} , Stride {2,2} , output units : 16, activation function activations[ACITVATION_INDEX],
                //RELU, for this case/
                //SubsamplingLayer stands for a Max Pooling layer, Kernel window size {3,3} , Stride {2,2}
                .layer(new ConvolutionLayer.Builder(3, 3)
                    .nIn(CHANNELS)
                    .stride(2, 2)
                    .nOut(16)
                    .activation(ACTIVATION_FNS[ACITVATION_INDEX])
                    .name("CONV LAYER 1")
                    .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(3, 3)
                    .stride(2, 2)
                    .build())

                //CONV LAYER 2: Kernel window size {3,3} , Stride {2,2} , output units : 32, activation function activations[ACITVATION_INDEX],
                //RELU, for this case/
                //SubsamplingLayer stands for a Max Pooling layer, Kernel window size {3,3} , Stride {2,2}
                .layer(new ConvolutionLayer.Builder(3, 3)
                    .stride(2, 2)
                    .nOut(32)
                    .activation(ACTIVATION_FNS[ACITVATION_INDEX])
                    .name("CONV LAYER 2")
                    .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(3, 3)
                    .stride(2, 2)
                    .build())

                //DENSE LAYER: output units : 32, activation function activations[ACITVATION_INDEX],
                //RELU, for this case/
                .layer(new DenseLayer.Builder().activation(ACTIVATION_FNS[ACITVATION_INDEX])
                    .nOut(64)
                    .build())


                //OUTPUT LAYER
                .layer(new OutputLayer.Builder(LOSS_FNS[LOSS_INDEX])
                    .nOut(OUTPUT_NUM)
                    .activation(Activation.SOFTMAX)
                    .build())
                // InputType.convolutional for normal image
                .setInputType(InputType.convolutionalFlat(HEIGHT_VALUES[SIZE_INDEX], WIDTH_VALUES[SIZE_INDEX], CHANNELS)).build();


            net = new MultiLayerNetwork(conf);
            net.init();

            //Sets the Training Listeners for the MultiLayerNetwork
            int listenerFrequency = 1;
            ArrayList<TrainingListener> listeners = new ArrayList<>();
            //Each 10 iterations, will show score performance on the console
            listeners.add(new ScoreIterationListener(10));

            //
            listeners.add(new StatsListener(statsStorage, listenerFrequency));
            net.setListeners(listeners);

            LOGGER.info("Total num of params: {}", net.numParams());

            /*Data Augmentation:
             *FlipImageTransform randomly flips the image according either the x-axis, y-axis, or both
             *WarpImageTransform warps the image either deterministically or randomly.
             * Thus, transformed images will still share the label of the original images
             * */

            ImageTransform warpTransform = new WarpImageTransform(randNumGen, 42);
            ImageTransform flipTransform = new FlipImageTransform(randNumGen);
            List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform, warpTransform});
            /* No data augmentation in this case ,List<ImageTransform>  created for further experiments  */

            // evaluation while training (the score should go down)
            for (int i = 0; i < EPOCHS; i++) {
                LOGGER.info("Starting epoch {} ...", i);
                trainRR.initialize(trainSplit);
                net.fit(trainIter);

                String pattern = "MMddyyyyHHmmss";
                SimpleDateFormat simpleDateFormat = new SimpleDateFormat(pattern);
                String date = simpleDateFormat.format(new Date());

                String modelFileLabel = date + "_" + ACTIVATION_FNS[ACITVATION_INDEX].name() + "_" + LOSS_FNS[LOSS_INDEX].name() + "_E" + i + "_model";

                createModelOutputFiles(foldDirectoryPath, modelFileLabel, testIter);

                trainIter.reset();
                testIter.reset();

                LOGGER.info("Completed epoch {}", i);

            }

        }

    }

    private static void createModelOutputFiles(String foldDirectoryPath, String modelFileLabel, DataSetIterator testIter) {

        String modelPath = foldDirectoryPath + "\\" + modelFileLabel + ".zip";
        Evaluation eval = net.evaluate(testIter);

        LOGGER.info(eval.stats());

        File modelZip = new File(modelPath);
        try {
            ModelSerializer.writeModel(net, modelZip, true);
        } catch (IOException e) {
            e.printStackTrace();
        }

        LOGGER.info("The model has been saved in {}", modelZip.getPath());
        String modelTextFilePath = foldDirectoryPath + "\\" + modelFileLabel + ".txt";

        try {
            FileWriter fw = new FileWriter(modelTextFilePath);
            BufferedWriter bw = new BufferedWriter(fw);
            bw.write(eval.stats());
            bw.newLine();
            bw.flush();

        } catch (IOException e) {
            LOGGER.error("modelTextFilePath FileWriter" + e.toString());
        }

        LOGGER.info("modelTextFilePath EXPORTED");
    }

    private static float calculateAccuracy(Evaluation eval) {

        float truePositives = (float) eval.getTruePositives().totalCount();
        float trueNegatives = (float) eval.getTrueNegatives().totalCount();

        float falsePositives = (float) eval.getFalsePositives().totalCount();
        float falseNegatives = (float) eval.getFalseNegatives().totalCount();

        float accuracy = (trueNegatives + truePositives) / (truePositives + falsePositives + trueNegatives + falseNegatives);
        return accuracy;
    }

    private static File checkCreateFile(String filePath) {

        LOGGER.info("checkCreateFile : " + filePath);

        File localFile = new File(filePath);

        if (!localFile.exists()) {
            LOGGER.info("File dont exist");
            localFile.mkdir();
        } else {
            LOGGER.info("File  exist");
        }

        return localFile;

    }


}
