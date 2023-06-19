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

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;

import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.CollectScoresIterationListener;
import org.deeplearning4j.optimize.listeners.CollectScoresListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
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
import org.nd4j.linalg.primitives.Pair;
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
 *Convolutional Neural Network implementation for crop type classification : "grape", "maize", "sunflower", "wheat"

 */
public class RetrainClassifier {



    private static MultiLayerNetwork net;

    private static final Logger LOGGER = LoggerFactory.getLogger(RetrainClassifier.class);
    private static final String SURFACE_TERMINAL_BASE_PATH = "E:\\02-17-20 Backup\\Documents\\IIASA";
    private static final String IIASA_TERMINAL_BASE_PATH = "H:\\MyDocuments";

    /*TERMINAL_BASE_PATH stores the root path for input image database directory and output models directory*/
    private static final String TERMINAL_BASE_PATH = SURFACE_TERMINAL_BASE_PATH;

    /*BASE_MODEL_DIR stores the path for output models directory*/
    private static final String BASE_MODEL_DIR = TERMINAL_BASE_PATH + "\\mxlTrainedModels";

    /*BASE_PATH stores the path for input Image Dataset directory*/
    private static final String BASE_PATH = TERMINAL_BASE_PATH + "\\Image Dataset";

    private static  int IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS,BATCH_SIZE ;
    private static int outputNum = 2; // 2 croptype classification

    private static String cropModelPrefix;

    private static ImagePreProcessingScaler trainImageScaler = new ImagePreProcessingScaler();

    private  static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();;

    public static void main(String[] args) throws Exception {
        String[] classNames1 = {"maize", "wheat"};
        String[] classNames2 = {"grape", "sunflower"};
        String[] classNames3 = {"grape", "maize", "sunflower", "wheat"};
        String[] cropModelTrainingPrefix = {"MW", "GS", "MWGS"};
        int cropTypeIndex = 0;//  cropModelTrainingPrefix = {"MW", "GS", "MWGS"}
        ArrayList<String[]> classNames = new ArrayList();
        classNames.add(classNames1);
        classNames.add(classNames2);
        classNames.add(classNames3);
        String[] classLabels;
        classLabels = classNames.get(cropTypeIndex);
        cropModelPrefix = cropModelTrainingPrefix[cropTypeIndex];


        //String modelLabel = "10222019235106_RELU_MSE_E14";
        String modelLabel = "04232020231606_MW_E2";

        String modelPath = TERMINAL_BASE_PATH + "\\mxlTrainedModels\\"+modelLabel+".zip";


        LOGGER.info("changes TEST");

        String[] cropModelTrainingPaths = {
            BASE_PATH + "\\" + cropModelTrainingPrefix[0],
            BASE_PATH + "\\" + cropModelTrainingPrefix[1],
            BASE_PATH + "\\" + cropModelTrainingPrefix[2]};


        int[] width = {100, 150, 200, 250, 300, 350, 400, 450, 500};
        int[] height = {56, 84, 112, 140, 169, 197, 225, 253, 281}; //width and height of the picture in px
        int sizeIndex = 0;

        int channels = 3;   // single channel for grayscale images

        int batchSize = 32; // number of samples that will be propagated through the network in each iteration
        int nEpochs = 15;    // number of training epochs
        int kfolds = 5; // number of times to train the experiment

        Random randNumGen = new Random();

        String localModelDirectoryPath = (BASE_MODEL_DIR + "\\" + cropModelTrainingPrefix[cropTypeIndex]);
        LOGGER.info("localModelDirectoryPath : " + localModelDirectoryPath);
        checkCreateFile(localModelDirectoryPath);

        LOGGER.info("Data vectorization...");
        // vectorization of train data
        String trainPath = cropModelTrainingPaths[cropTypeIndex] + "\\train";
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

        String testPath = cropModelTrainingPaths[cropTypeIndex] + "\\test";
        File testData = new File(testPath);
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        ImageRecordReader testRR = new ImageRecordReader(height[sizeIndex], width[sizeIndex], channels, labelMaker);
        testRR.initialize(testSplit);

        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
        testIter.setPreProcessor(imageScaler); // same ImageRecordReader normalization for better results

        LOGGER.info("Network configuration and training...");

        /*MultiLayerConfiguration sets the CNN arquitecture configuration for training,
         * in this case, consists in 2 convolution layers, then a fully connected layer and finally an output layer*/

        LOGGER.info("modelPath: " + modelPath);

        LOGGER.info("Gets MultiLayerConfiguration ...");


        MultiLayerNetwork multiLayerNetwork;

        try {
            multiLayerNetwork = getModelNetwork(modelPath);
            Updater updater =  multiLayerNetwork.getUpdater();

            LOGGER.info("Network configuration and training...");
            // reduce the learning rate as the number of training epochs increases
            // iteration #, learning rate
            Map<Integer, Double> learningRateSchedule = new HashMap<>();
            learningRateSchedule.put(0, 0.06);
            learningRateSchedule.put(200, 0.05);
            learningRateSchedule.put(600, 0.028);
            learningRateSchedule.put(800, 0.006);
            learningRateSchedule.put(1000, 0.001);
            //values={0=0.06, 800=0.006, 200=0.05, 600=0.028, 1000=0.001}

            /*MultiLayerConfiguration sets the CNN arquitecture configuration for training,
             * in this case, consists in 2 convolution layers, then a fully connected layer and finally an output layer*/
            multiLayerNetwork.getDefaultConfiguration().clearVariables();
            multiLayerNetwork.clearNoiseWeightParams();
            multiLayerNetwork.clearLayersStates();
            multiLayerNetwork.clearLayerMaskArrays();
            multiLayerNetwork.clear();
            multiLayerNetwork.getDefaultConfiguration().setIterationCount(0);
            multiLayerNetwork.getDefaultConfiguration().setEpochCount(0);


            long networksSeed = multiLayerNetwork.getDefaultConfiguration().getSeed();
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
                .seed(networksSeed)
                .weightInit(WeightInit.XAVIER)
                .list()

                //CONV LAYER 1: Kernel window size {3,3} , Stride {2,2} , output units : 16, activation function activations[iActivation],
                //RELU, for this case/
                //SubsamplingLayer stands for a Max Pooling layer, Kernel window size {3,3} , Stride {2,2}
                .layer(new ConvolutionLayer.Builder(3, 3)
                    .weightInit(WeightInit.XAVIER)
                    .nIn(channels)
                    .stride(2, 2)
                    .padding(0,0)
                    .nOut(16)
                    .activation(Activation.RELU)
                    .name("CONV LAYER 1")
                    .build())

                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(3, 3)
                    .stride(2, 2)
                    .padding(0,0)
                    .name("MAX POOLING 1")
                    .build())

                //CONV LAYER 2: Kernel window size {3,3} , Stride {2,2} , output units : 32, activation function activations[iActivation],
                //RELU, for this case/
                //SubsamplingLayer stands for a Max Pooling layer, Kernel window size {3,3} , Stride {2,2}
                .layer(new ConvolutionLayer.Builder(3, 3)
                    .weightInit(WeightInit.XAVIER)
                    .stride(2, 2)
                    .nOut(32)
                    .padding(0,0)
                    .activation(Activation.RELU)
                    .name("CONV LAYER 2")
                    .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(3, 3)
                    .stride(2, 2)
                    .name("MAX POOLING 2")
                    .build())

                //DENSE LAYER: output units : 32, activation function activations[iActivation],
                //RELU, for this case/
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                    .nOut(512)
                    .name("DENSE LAYER")
                    .build())


                //OUTPUT LAYER
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .weightInit(WeightInit.XAVIER)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .name("OUTPUT LAYER")
                    .build())
                // InputType.convolutional for normal image
                .setInputType(InputType.convolutionalFlat(height[sizeIndex], width[sizeIndex], channels)) .build();

            net = new MultiLayerNetwork(conf);
            net.init();


            //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
            //Then add the StatsListener to collect this information from the network, as it trains

            String pattern = "MMddyyyyHHmmss";
            SimpleDateFormat simpleDateFormat = new SimpleDateFormat(pattern);
            String date = simpleDateFormat.format(new Date());

            StatsStorage statsStorage = new FileStatsStorage(new File(BASE_MODEL_DIR + "/"+ date +"_dl4jstats.dl4j"));

            //Initialize the user interface backend
            UIServer uiServer = UIServer.getInstance();

            //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
            uiServer.attach(statsStorage);


            //StatsStorage statsStorage = new FileStatsStorage(statsFile);    //If file already exists: load the data from it
            //    UIServer uiServer = UIServer.getInstance();
            //    uiServer.attach(statsStorage);


            //Sets the Training Listeners for the MultiLayerNetwork
            int listenerFrequency = 1;
            ArrayList<TrainingListener> listeners = new ArrayList<>();
            //Each 10 iterations, will show score performance on the console
            //Score iteration listener. Reports the score (value of the loss function )of the network during training every N iterations
            listeners.add(new ScoreIterationListener(listenerFrequency));
            //Simple IterationListener that tracks time spend on training per iteration.
            listeners.add(new PerformanceListener(listenerFrequency));
            //StatsListener: a general purpose listener for collecting and reporting system and model information.
            listeners.add(new StatsListener(statsStorage, listenerFrequency));

            CollectScoresIterationListener collectScoreListener = new CollectScoresIterationListener(listenerFrequency);
            String scoresPath = BASE_MODEL_DIR + "\\"+ date + "_scores.csv";
            listeners.add(collectScoreListener);

            multiLayerNetwork.setListeners(listeners);
            multiLayerNetwork.fit(trainIter,nEpochs);
            LOGGER.info("multiLayerNetwork.fit(trainIter,nEpochs) DONE");
//
//            date = simpleDateFormat.format(new Date());
//            String modelFileLabel = date + "_" + cropModelTrainingPrefix[cropTypeIndex] + "_ALL_EPOCHS";
//            LOGGER.info(" multiLayerNetwork modelFileLabel :" + modelFileLabel);
//            createModelOutputFiles(multiLayerNetwork, localModelDirectoryPath, modelFileLabel , testIter);

            net.setListeners(listeners);
            net.fit(trainIter,nEpochs);
            LOGGER.info("net.fit(trainIter,nEpochs) DONE");

            date = simpleDateFormat.format(new Date());
            String modelFileLabel = date + "_" + cropModelTrainingPrefix[cropTypeIndex] + "_ALL_EPOCHS";
            LOGGER.info("net modelFileLabel :", modelFileLabel);
            createModelOutputFiles(net, localModelDirectoryPath, modelFileLabel , testIter);

            net = new MultiLayerNetwork(conf);
            net.setListeners(listeners);
            net.init();


            LOGGER.info("trainRR.initialize(trainSplit) + net.fit(trainIter) BEGINS");
            date = simpleDateFormat.format(new Date());
            // evaluation while training (the score should go down)
            for (int i = 0; i < nEpochs; i++) {

                trainRR.initialize(trainSplit);
                net.fit(trainIter);


                modelFileLabel = date + "_" + cropModelTrainingPrefix[cropTypeIndex] + "_E" + i ;
                LOGGER.info("trainRR.initialize(trainSplit) modelFileLabel :", modelFileLabel);
                createModelOutputFiles(net, localModelDirectoryPath, modelFileLabel , testIter);

                trainIter.reset();
                testIter.reset();

                scoresPath = BASE_MODEL_DIR + "\\"+ date + "_E" + i + "_scores.csv";
                collectScoreListener.exportScores(new File(scoresPath),",");

                LOGGER.info("Completed epoch {}", i);

            }

            collectScoreListener.exportScores(new File(scoresPath),",");

            LOGGER.info("Epochs completed");


        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    private static void createModelOutputFiles(MultiLayerNetwork network , String foldDirectoryPath,  String modelFileLabel , DataSetIterator testIter) {

        String modelPath = foldDirectoryPath + "\\" + modelFileLabel + ".zip";
        Evaluation eval = network.evaluate(testIter);

        LOGGER.info(eval.stats());

        File modelZip = new File(modelPath);

        try {
            ModelSerializer.writeModel(network, modelZip, true);
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

    private static MultiLayerNetwork getModelNetwork(String modelPath) throws IOException {

        MultiLayerNetwork result = null;
        File model = new File(modelPath);

        if (!model.exists()) {
            throw new IOException("Can't find the model");
        } else {
            LOGGER.info("model file exists");
        }

        result = ModelSerializer.restoreMultiLayerNetwork(model);


        initSampleInputParams(result);
        //showEvalStats(result);

        return  result;

    }


    private static void initSampleInputParams(MultiLayerNetwork multiLayerNetwork) {
        FeedForwardToCnnPreProcessor preProcessor = (FeedForwardToCnnPreProcessor) multiLayerNetwork.getLayerWiseConfigurations().getInputPreProcessors().get(0);
        IMAGE_HEIGHT = (int) preProcessor.getInputHeight();
        IMAGE_WIDTH = (int) preProcessor.getInputWidth();
        IMAGE_CHANNELS = (int) preProcessor.getNumChannels();
        BATCH_SIZE = 100;
        //net.setConf(net.getLayerWiseConfigurations().getConfs().get(0));

        LOGGER.info("inputHeight : " + IMAGE_HEIGHT);
        LOGGER.info("inputWidth : " + IMAGE_WIDTH);
        LOGGER.info("inputChannels : " + IMAGE_CHANNELS);

        String trainDatPath = BASE_PATH + "\\"+cropModelPrefix+"\\train";
        File trainData = new File(trainDatPath);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // use parent directory name as the image label
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, new Random(1234));

        ImageRecordReader trainRR = new ImageRecordReader(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, labelMaker);

        try {
            trainRR.initialize(trainSplit);
        } catch (IOException e) {
            e.printStackTrace();
        }

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, BATCH_SIZE, 1, outputNum);

        // pixel values from 0-255 to 0-1 (min-max scaling)
        trainImageScaler = new ImagePreProcessingScaler();
        trainImageScaler.fit(trainIter);

    }
    private static  void showEvalStats(MultiLayerNetwork multiLayerNetwork) {
        // vectorization of test data

        labelMaker = new ParentPathLabelGenerator();

        File testData = new File(BASE_PATH + "\\"+cropModelPrefix+ "\\test");
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS,  new Random());
        ImageRecordReader testRR = new ImageRecordReader(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, labelMaker);

        try {
            testRR.initialize(testSplit);
        } catch (IOException e) {
            e.printStackTrace();
        }

        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, BATCH_SIZE, 1, outputNum);
        testIter.setPreProcessor(trainImageScaler); // same ImageRecordReader normalization for better results
        Evaluation eval = multiLayerNetwork.evaluate(testIter);
        LOGGER.info("showEvalStats()");
        LOGGER.info(eval.stats());

    }

}
