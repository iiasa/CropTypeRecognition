package org.deeplearning4j.examples.convolution;

//import javafx.scene.control.Label;
//import javafx.scene.image.ImageView;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.text.html.ImageView;
import java.awt.image.BufferedImage;
import java.io.*;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Random;


/**
 * Test UI for Crop classifier obtained from transfer learning.  <br>
 * Run the {@link org.deeplearning4j.examples.vgg16.FeaturizedPreSave} first to build the model.
 *
 * @author marcial sg
 */



public class CropModelEnsambleFromTerminalv3 {

    static DataNormalization trainImageScaler;
    private static final Logger LOGGER = LoggerFactory.getLogger(CropModelEnsambleFromTerminalv3.class);

    private static final String SURFACE_TERMINAL_BASE_PATH = "E:\\02-17-20 Backup\\Documents\\IIASA";
    private static final String IIASA_TERMINAL_BASE_PATH = "E:\\titan_documents\\IIASA";
    private static final String TITAN_TERMINAL_BASE_PATH = "C:\\Users\\HP WorkStation\\Documents\\Documentos Titan\\IIASA";
    private static final String TERMINAL_BASE_PATH = IIASA_TERMINAL_BASE_PATH;
    private static final String BASE_PATH = TERMINAL_BASE_PATH + "\\Image Dataset";
    private static final String PICTURE_PILE_PATH = TERMINAL_BASE_PATH + "\\earthchallengeImages";
    private static final String PICTURES_VAL_PATH = PICTURE_PILE_PATH + "\\crop_type_v4_data\\mwo\\val";

    static private String newline = "\n";
    private static int BATCH_SIZE = 100;



    private static int IMAGE_WIDTH;
    private static int IMAGE_HEIGHT;
    private static int IMAGE_CHANNELS;
    static int outputNum = 5; // 2 croptype classification
    // array of supported extensions (use a List if you prefer)
    static final String[] EXTENSIONS = new String[]{
        "gif", "png", "bmp", "jpg"// and other formats you need
    };
    JFileChooser fc = null, modelChooser = null;
    ImageView imgView;

    static String[]  classNames1 = {"maize", "wheat"};
    static String[] classNames2 = {"grape", "sunflower"};
    static String[] classNames3 = {"grape", "maize", "sunflower", "wheat"};
    static String[] classNames4 = {"daisy", "dandelion", "roses", "sunflowers", "tulips"};
    static String[] classNames5 = {"maize", "other", "wheat"};
    static String[] cropModelTrainingPrefix = {"MW", "GS", "MWGS", "FLOWERS", "MWO"};
    static String[] classLabels;
    static String cropModelPrefix;
    static int cropTypeIndex = 4;//  MWO

    int PICTURE_PILE_BATCH_NO = 2;

    String[] cropModelTrainingPaths = {
        BASE_PATH + "\\" + cropModelTrainingPrefix[0],
        BASE_PATH + "\\" + cropModelTrainingPrefix[1],
        BASE_PATH + "\\" + cropModelTrainingPrefix[2],
        BASE_PATH + "\\" + cropModelTrainingPrefix[3],
        BASE_PATH + "\\" + cropModelTrainingPrefix[4]};

    static ArrayList<String[]> classNames = new ArrayList();

//    private final double[] maizeCentroid = {0.965910816,	0.520882138,	0.013386066,	0.977219936};
//    private final double[] otherCentroid = {0.489447644,	0.826917604,	0.625937269,	0.818640766};
//    private final double[] wheatCentroid =  {0.815427272,	0.531788577,	0.939501779,	0.949467153};
//    private final double[][] centroids = {maizeCentroid, otherCentroid ,wheatCentroid};

    final int[] resnetMixMaxValues = {0,998};

    static MultiLayerNetwork multiLayerNetwork;
    static ComputationGraph vgg16Model; // base model
    static ComputationGraph resnet50Model; // base model


    // filter to identify images based on their extensions
    static final FilenameFilter IMAGE_FILTER = new FilenameFilter() {

        @Override
        public boolean accept(final File dir, final String name) {

            if (dir.isDirectory() && !(name.endsWith(".ini"))  && !(name.endsWith(".json"))  && !(name.endsWith(".db") && !(name.endsWith(".json")))) {
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

    DataSetIterator testIter;
    FileSplit testSplit;
    ParentPathLabelGenerator labelMaker;
    static private int TLMODEL_IMAGE_HEIGHT;
    static private int TLMODEL_IMAGE_WIDTH;
    static private final int VGG16_CODE = 100;
    static private final int MW_CODE = 200;
    static private final int RESNET50_CODE = 300;
    static private String testResult;

    private static int BATCH_INIT = 0;
    private static int BATCH_END = 26;

    public static void main(String[] args) {

        classNames.add(classNames1);
        classNames.add(classNames2);
        classNames.add(classNames3);
        classNames.add(classNames4);
        classNames.add(classNames5);

        classLabels = classNames.get(cropTypeIndex);
        cropModelPrefix = cropModelTrainingPrefix[cropTypeIndex];

        LOGGER.info("");

        IMAGE_CHANNELS = 3;
        TLMODEL_IMAGE_HEIGHT = 224;
        TLMODEL_IMAGE_WIDTH = 224;

        String resnet50ModelPath = TERMINAL_BASE_PATH + "\\Transfer Learning models\\crops\\resnet50.zip";
        String mwModelPath = TERMINAL_BASE_PATH + "\\earthchallengeImages\\models\\mwo\\2023\\K1\\03032023152021_RELU_MSE_E14_model.zip";

        try {
            setPreTrainedModelModel(resnet50ModelPath, RESNET50_CODE);
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            setModel(mwModelPath);
        } catch (IOException e) {
            e.printStackTrace();
        }

        createPicturePileCSV();


    }

    static private void initSampleInputParams() {

        FeedForwardToCnnPreProcessor preProcessor = (FeedForwardToCnnPreProcessor) multiLayerNetwork.getLayerWiseConfigurations().getInputPreProcessors().get(0);
        IMAGE_HEIGHT = (int) preProcessor.getInputHeight();
        IMAGE_WIDTH = (int) preProcessor.getInputWidth();

        String trainDatPath = BASE_PATH + "\\" + cropModelPrefix + "\\trn";
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
        DataNormalization imageScaler = new ImagePreProcessingScaler();
        imageScaler.fit(trainIter);

        trainIter.setPreProcessor(imageScaler);

        // pixel values from 0-255 to 0-1 (min-max scaling)
        trainImageScaler = new ImagePreProcessingScaler();
        trainImageScaler.fit(trainIter);

    }

    static private void setModel(String modelPath) throws IOException {

        File model = new File(modelPath);

        if (!model.exists()) {
            throw new IOException("Can't find the model");
        } else {
            LOGGER.info("model file exists");
        }

        multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(model);

        initSampleInputParams();
        // Evaluation evaluation = showEvalStats();
        //showROCStats(evaluation);

    }

    static private void setPreTrainedModelModel(String modelPath, int modelCode) throws IOException {


        File model = new File(modelPath);

        if (!model.exists()) {
            throw new IOException("Can't find the model");
        } else {
            LOGGER.info("model file exists");
        }

        switch (modelCode) {
            case RESNET50_CODE:
                resnet50Model = ComputationGraph.load(model, true);
                break;
            case VGG16_CODE:
                vgg16Model = ComputationGraph.load(model, true);
                break;
        }

        // Evaluation evaluation = showEvalStats();
        //showROCStats(evaluation);

    }

    /**
     * This method classifies the input image.
     *
     * @param  newImagePath Filepath with the test image to be classified.
     * @return String[] class label on index 0, and confidence value on index 1.
     */
    static private String[] testImage(String newImagePath, int modelCode) {

        String[] result = new String[2];
        String prediction = "", probability = "";

        int predictionClass;
        double probabilityValue;

        switch (modelCode) {

            case RESNET50_CODE:
                try {
                    NativeImageLoader loader = new NativeImageLoader(TLMODEL_IMAGE_HEIGHT, TLMODEL_IMAGE_WIDTH, IMAGE_CHANNELS);
                    INDArray image = loader.asMatrix(new File(newImagePath));

                    VGG16ImagePreProcessor imagePreProcessor = new VGG16ImagePreProcessor();
                    imagePreProcessor.transform(image);

                    INDArray[] samples = {image};
                    INDArray[] outputPrediction = resnet50Model.output(false, samples);

                    predictionClass = (int) outputPrediction[0].argMax(1).getDouble(0);
                    probabilityValue = outputPrediction[0].getDouble(predictionClass);

                    prediction = String.valueOf(predictionClass);
                    probability = String.valueOf(probabilityValue);

                } catch (IOException e) {
                    e.printStackTrace();
                }

                break;

            default:

                NativeImageLoader loader = new NativeImageLoader(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS);

                try {

                    INDArray image = loader.asMatrix(new File(newImagePath));
                    trainImageScaler.transform(image);

                    int[] outputPrediction = multiLayerNetwork.predict(image);
                    prediction = String.valueOf(outputPrediction[0]);

                    INDArray output = multiLayerNetwork.output(image);
                    probability = String.valueOf(output.getDouble(outputPrediction[0]));

                } catch (IOException e) {
                    e.printStackTrace();
                }

                break;
        }


        result[0] = prediction;
        result[1] = probability;

        return result;
    }

    static private void createPicturePileCSV() {

//        String imagesPath = PICTURE_PILE_BATCHES_PATH + "\\batch" + batch;
        String imagesPath = "E:\\titan_documents\\IIASA\\earthchallengeImages\\crop_type_v4_data\\mwo\\val";

        File dir = new File(imagesPath);

        String outputDataset = "";
        String titles = "image_name" + "," + "label" + "," + "tf_model_label" + "," + "tf_model_value" + "," + "mwo_model_label" + "," + "mwo_model_value" + "\n";
        outputDataset += titles;


        for (final File classFolder : dir.listFiles()){

            System.out.println("folder: " + classFolder.getName());

            for (final File imgFile : classFolder.listFiles(IMAGE_FILTER)) {

                BufferedImage img = null;
                String imgLabel = imgFile.getName().split("_")[1];

                try {

                    img = ImageIO.read(imgFile);
                    System.out.println("image : " + imgFile.getName());
                    System.out.println("imgLabel: " + imgLabel);


                    if (img != null) {

                        String[] output1 = testImage(imgFile.getPath(), RESNET50_CODE);
                        String[] output2 = testImage(imgFile.getPath(), MW_CODE);

                        System.out.println("RESTNET50: " + output1[0]);
                        System.out.println("MWO: " + output2[0]);

                        DecimalFormat df = new DecimalFormat("0.000000"); //TODO: CHECK IF 6 FLOATING POINTS ARE ENOUGH

                        String sample = imgFile.getName() + "," + imgLabel + "," + output1[0] + "," + df.format(Double.parseDouble(output1[1])) + "," + output2[0] + "," + df.format(Double.parseDouble(output2[1])) + "\n";
                        outputDataset += sample;

                    }

                } catch (final IOException e) {
                    // handle errors here
                }

            }

        }

        saveDatasetToCSV(outputDataset);

        LOGGER.info("CSV File Created");
    }

    static private void saveDatasetToCSV(String outputDataset) {

        String pattern = "MMddyyyyHHmmss";
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat(pattern);
        String date = simpleDateFormat.format(new Date());
        String csvPath = TERMINAL_BASE_PATH + "\\cropModelEnsemble_2023Results\\cropModelVal_"+ date +".csv";

        try {
            FileWriter fw = new FileWriter(csvPath,true);
            BufferedWriter bw = new BufferedWriter(fw);
            bw.write(outputDataset);
            bw.flush();

        } catch (IOException e) {
            System.out.println("FileWriter IOException: " + e.toString());
        }


    }

}
