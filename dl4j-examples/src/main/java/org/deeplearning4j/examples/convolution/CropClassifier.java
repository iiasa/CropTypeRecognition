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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Convolutional Neural Network implementation for crop type classification : "grape", "maize", "sunflower", "wheat"
 */
public class CropClassifier {


    private static MultiLayerNetwork net;

    private static final Logger LOGGER = LoggerFactory.getLogger(CropClassifier.class);
    private static final String SURFACE_TERMINAL_BASE_PATH = "E:\\02-17-20 Backup\\Documents\\IIASA";
    private static final String IIASA_TERMINAL_BASE_PATH = "H:\\MyDocuments";
    private static final String TITAN_TERMINAL_BASE_PATH = "C:\\Users\\HP WorkStation\\Documents\\IIASA";

    /*TERMINAL_BASE_PATH stores the root path for input image database directory and output models directory*/
    private static final String TERMINAL_BASE_PATH = TITAN_TERMINAL_BASE_PATH;

    /*BASE_MODEL_DIR stores the path for output models directory*/
    private static final String BASE_MODEL_DIR = TERMINAL_BASE_PATH + "\\mxlTrainedModels";


    /*BASE_PATH stores the path for input Image Dataset directory*/
    private static final String BASE_PATH = TERMINAL_BASE_PATH + "\\earthchallengeImages\\crop_type_v2_data";


    public static void main(String[] args) throws Exception {
        String[] classNames1 = {"maize", "wheat"};
        String[] classNames2 = {"grape", "sunflower"};
        String[] classNames3 = {"grape", "maize", "sunflower", "wheat"};
        String[] cropModelTrainingPrefix = {"MW", "GS", "MWGS"};

        int cropTypeIndex = 0;//  cropModelTrainingPrefix = {"MW", "GS", "MWGS"};

        ArrayList<String[]> classNames = new ArrayList();
        classNames.add(classNames1);
        classNames.add(classNames2);
        classNames.add(classNames3);
        String[] classLabels;
        String cropModelPrefix;
        classLabels = classNames.get(cropTypeIndex);
        cropModelPrefix = cropModelTrainingPrefix[cropTypeIndex];


        String modelLabel = "10222019235106_RELU_MSE_E14";

        String modelPath = TERMINAL_BASE_PATH + "\\best models\\" + cropModelPrefix + "\\" + modelLabel + "\\" + modelLabel + "_model.zip";


        LOGGER.info("changes TEST");

        String[] cropModelTrainingPaths = {
            BASE_PATH + "\\" + cropModelTrainingPrefix[0],
            BASE_PATH + "\\" + cropModelTrainingPrefix[1],
            BASE_PATH + "\\" + cropModelTrainingPrefix[2]};

        int[] width = {100, 150, 200, 250, 300, 350, 400, 450, 500};
        int[] height = {56, 84, 112, 140, 169, 197, 225, 253, 281}; //width and height of the picture in px
        int sizeIndex = 0;

        int channels = 3;   // single channel for grayscale images
        int outputNum = 2; // 2 croptype classification
        int batchSize = 100; // number of samples that will be propagated through the network in each iteration
        int nEpochs = 15;    // number of training epochs
        int kfolds = 5; // number of times to train the experiment

        Random randNumGen = new Random();

        /*Possible activation function options*/
        Activation[] activations = {Activation.RELU, Activation.IDENTITY, Activation.TANH}; // Activation.SIGMOID CRASHED
        int iActivation = 0;

        /*Possible loss function options for output layer*/
        LossFunctions.LossFunction[] lossFunctions = {LossFunctions.LossFunction.MSE, LossFunctions.LossFunction.POISSON, LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR};
        int iLossFuction = 0;

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "cropclassifier_stats.dl4j"));

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        for (int cropType = 0; cropType < classNames.size(); cropType++) {

            String localModelDirectoryPath = (BASE_MODEL_DIR + "\\" + cropModelTrainingPrefix[cropType]);
            LOGGER.info("localModelDirectoryPath : " + localModelDirectoryPath);
            checkCreateFile(localModelDirectoryPath);

            for (int k = 0; k < kfolds; k++) {

                /*Creates a directory for each iteration of the experiment*/
                String foldDirectoryPath = (localModelDirectoryPath + "\\K" + k);
                checkCreateFile(foldDirectoryPath);

                LOGGER.info("Data vectorization...");
                // vectorization of train data
                String trainPath = cropModelTrainingPaths[cropType] + "\\train";
                File trainData = new File(trainPath);

                /*Creates Train and Test Dataset iterator for further model training*/
                ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // use parent directory name as the image label

                ImageRecordReader trainRR = new ImageRecordReader(height[sizeIndex], width[sizeIndex], channels, labelMaker);
                FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
                trainRR.initialize(trainSplit);

                DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

                // pixel values from 0-255 to 0-1 (min-max scaling)
                DataNormalization imageScaler = new ImagePreProcessingScaler();
                imageScaler.fit(trainIter);

                trainIter.setPreProcessor(imageScaler);

                File testData = new File(cropModelTrainingPaths[cropType] + "\\test");
                FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
                ImageRecordReader testRR = new ImageRecordReader(height[sizeIndex], width[sizeIndex], channels, labelMaker);
                testRR.initialize(testSplit);

                DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
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

                    //CONV LAYER 1: Kernel window size {3,3} , Stride {2,2} , output units : 16, activation function activations[iActivation],
                    //RELU, for this case/
                    //SubsamplingLayer stands for a Max Pooling layer, Kernel window size {3,3} , Stride {2,2}
                    .layer(new ConvolutionLayer.Builder(3, 3)
                        .nIn(channels)
                        .stride(2, 2)
                        .nOut(16)
                        .activation(activations[iActivation])
                        .name("CONV LAYER 1")
                        .build())
                    .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build())

                    //CONV LAYER 2: Kernel window size {3,3} , Stride {2,2} , output units : 32, activation function activations[iActivation],
                    //RELU, for this case/
                    //SubsamplingLayer stands for a Max Pooling layer, Kernel window size {3,3} , Stride {2,2}
                    .layer(new ConvolutionLayer.Builder(3, 3)
                        .stride(2, 2)
                        .nOut(32)
                        .activation(activations[iActivation])
                        .name("CONV LAYER 2")
                        .build())
                    .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build())

                    //DENSE LAYER: output units : 32, activation function activations[iActivation],
                    //RELU, for this case/
                    .layer(new DenseLayer.Builder().activation(activations[iActivation])
                        .nOut(64)
                        .build())


                    //OUTPUT LAYER
                    .layer(new OutputLayer.Builder(lossFunctions[iLossFuction])
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                    // InputType.convolutional for normal image
                    .setInputType(InputType.convolutionalFlat(height[sizeIndex], width[sizeIndex], channels)).build();


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
                for (int i = 0; i < nEpochs; i++) {

                    trainRR.initialize(trainSplit);
                    net.fit(trainIter);

                    String pattern = "MMddyyyyHHmmss";
                    SimpleDateFormat simpleDateFormat = new SimpleDateFormat(pattern);
                    String date = simpleDateFormat.format(new Date());

                    String modelFileLabel = date + "_" + activations[iActivation].name() + "_" + lossFunctions[iLossFuction].name() + "_E" + i + "_model";

                    createModelOutputFiles(foldDirectoryPath, modelFileLabel, testIter);

                    trainIter.reset();
                    testIter.reset();

                    LOGGER.info("Completed epoch {}", i);

                }

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
