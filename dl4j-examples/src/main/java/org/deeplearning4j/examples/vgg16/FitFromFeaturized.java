/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

package org.deeplearning4j.examples.vgg16;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.vgg16.dataHelpers.CropDataSetIteratorFeaturized;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

//import org.nd4j.jita.conf.CudaEnvironment;

/**
 * @author susaneraly on 3/10/17.
 * <p>
 * Important:
 * Run the class "FeaturizePreSave" before attempting to run this. The outputs at the boundary of the frozen and unfrozen
 * vertices of a model are saved. These are referred to as "featurized" datasets in this description.
 * On a dataset of about 3000 images which is what is downloaded this can take "a while"
 * <p>
 * Here we see how the transfer learning helper can be used to fit from a featurized datasets.
 * We attempt to train the same model architecture as the one in "EditLastLayerOthersFrozen".
 * Since the helper avoids the forward pass through the frozen layers we save on computation time when running multiple epochs.
 * In this manner, users can iterate quickly tweaking learning rates, weight initialization etc` to settle on a model that gives good results.
 */
public class FitFromFeaturized {

    public static final String featureExtractionLayer = FeaturizedPreSave.featurizeExtractionLayer;
    protected static final long seed = 12345;
    protected static final int numClasses = 3;
    protected static final int nEpochs = 28;
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FitFromFeaturized.class);
    private static final String SURFACE_TERMINAL_BASE_PATH = "E:\\02-17-20 Backup\\Documents\\IIASA";
    private static final String IIASA_TERMINAL_BASE_PATH = "H:\\MyDocuments";
    private static final String TITAN_TERMINAL_BASE_PATH = "C:\\Users\\HP WorkStation\\Documents\\IIASA";
    /*TERMINAL_BASE_PATH stores the root path for input image database directory and output models directory*/
    private static final String TERMINAL_BASE_PATH = TITAN_TERMINAL_BASE_PATH;


    private static final String BASE_PATH = TERMINAL_BASE_PATH + "\\Image Dataset";
    private static ImagePreProcessingScaler trainImageScaler;

    private static ParentPathLabelGenerator labelMaker;
    private static long IMAGE_HEIGHT = 224;
    private static long IMAGE_WIDTH = 224;
    private static long IMAGE_CHANNELS = 3;
    private static String cropModelPrefix = "MWO";
    private static int BATCH_SIZE = 32;
    private static int outputNum = 3;
    private static RecordReaderDataSetIterator testIter;

    public static void main(String[] args) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {

        // temp workaround for backend initialization

//        CudaEnvironment.getInstance().getConfiguration()
//            // key option enabled
//            .allowMultiGPU(true)
//
//            // we're allowing larger memory caches
//            .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)
//
//            // cross-device access is used for faster model averaging over pcie
//            .allowCrossDeviceAccess(true);

        //Import vgg
        //Note that the model imported does not have an output layer (check printed summary)
        //  nor any training related configs (model from keras was imported with only weights and json)
        log.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
//        ZooModel zooModel = VGG16.builder().build();
        ZooModel resNet50Model = ResNet50.builder().build();

        ComputationGraph pretrainedNet = (ComputationGraph) resNet50Model.initPretrained(PretrainedType.IMAGENET);
        log.info(pretrainedNet.summary());

        savePretrainedModel(pretrainedNet);

        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(5e-5))
            .seed(seed)
            .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrainedNet)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor(featureExtractionLayer) //the specified layer and below are "frozen"
            .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
            .addLayer("predictions",
                new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(4096).nOut(numClasses)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.SOFTMAX).build(),
                featureExtractionLayer)
            .build();
        log.info(vgg16Transfer.summary());

        DataSetIterator trainIter = CropDataSetIteratorFeaturized.trainIterator();
        DataSetIterator testIter = CropDataSetIteratorFeaturized.testIterator();

        System.out.println("Env information " + Nd4j.getExecutioner().getEnvironmentInformation());

        //Instantiate the transfer learning helper to fit and output from the featurized dataset
        //The .unfrozenGraph() is the unfrozen subset of the computation graph passed in.
        //If using with a UI or a listener attach them directly to the unfrozenGraph instance
        //With each iteration updated params from unfrozenGraph are copied over to the original model
        TransferLearningHelper transferLearningHelper = new TransferLearningHelper(vgg16Transfer);
        log.info(transferLearningHelper.unfrozenGraph().summary());

        for (int epoch = 0; epoch < nEpochs; epoch++) {
            transferLearningHelper.unfrozenGraph().fit(trainIter);
            trainIter.reset();
            log.info("Epoch #" + epoch + " complete");
        }

        INDArray trainedParamW = transferLearningHelper.unfrozenGraph().getLayer("predictions").getParam("W");
        INDArray trainedParamB = transferLearningHelper.unfrozenGraph().getLayer("predictions").getParam("b");
        vgg16Transfer.getLayer("predictions").setParam("W", trainedParamW);
        vgg16Transfer.getLayer("predictions").setParam("b", trainedParamB);

        log.info("Model build complete");

        showEvalStats(vgg16Transfer);

        String pattern = "MMddyyyyHHmmss";
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat(pattern);
        String date = simpleDateFormat.format(new Date());

        String tlFolderPath = TITAN_TERMINAL_BASE_PATH + "\\Transfer Learning models\\crops";
        File tlFolder = checkCreateFile(tlFolderPath);

        String modelPath = tlFolder.getPath() + "\\" + date + ".zip";
        File modelZip = new File(modelPath);

        try {

            vgg16Transfer.save(modelZip);

            // ModelSerializer.writeModel(transferLearningHelper.unfrozenGraph(), modelZip, true);
            log.info("The model has been saved in {}", modelZip.getPath());

        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    private static void savePretrainedModel(ComputationGraph pretrainedNet) {
        String resnetFolderPath = TITAN_TERMINAL_BASE_PATH + "\\Transfer Learning models\\crops";
        File vFolder = checkCreateFile(resnetFolderPath);

        String resnet50ModelPath = vFolder.getPath() + "\\resnet50.zip";
        File resnet50ModelZip = new File(resnet50ModelPath);

        try {

            pretrainedNet.save(resnet50ModelZip);

            // ModelSerializer.writeModel(transferLearningHelper.unfrozenGraph(), modelZip, true);
            log.info("The model has been saved in {}", resnet50ModelZip.getPath());

        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    private static File checkCreateFile(String filePath) {

        log.info("checkCreateFile : " + filePath);

        File localFile = new File(filePath);

        if (!localFile.exists()) {
            log.info("File dont exist");
            localFile.mkdir();
        } else {
            log.info("File  exist");
        }

        return localFile;

    }

    private static Evaluation showEvalStats(ComputationGraph model) {

        labelMaker = new ParentPathLabelGenerator();
        String testPath = BASE_PATH + "\\" + cropModelPrefix + "\\test";
        File testData = new File(testPath);
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, new Random());
        ImageRecordReader testRR = new ImageRecordReader(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, labelMaker);

        try {
            testRR.initialize(testSplit);
        } catch (IOException e) {
            e.printStackTrace();
        }

        testIter = new RecordReaderDataSetIterator(testRR, BATCH_SIZE, 1, outputNum);
        testIter.setPreProcessor(new VGG16ImagePreProcessor());
        Evaluation eval = model.evaluate(testIter);
        System.out.println(eval.stats());
        return eval;

    }


}
