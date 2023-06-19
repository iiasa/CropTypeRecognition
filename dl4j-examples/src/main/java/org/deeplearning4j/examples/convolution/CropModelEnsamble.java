//package org.deeplearning4j.examples.convolution;
//
//import java.awt.*;
//import java.awt.image.BufferedImage;
//import java.io.*;
//import java.text.DecimalFormat;
//import java.text.SimpleDateFormat;
//import java.util.ArrayList;
//import java.util.Date;
//import java.util.Map;
//import java.util.Random;
//
//import javafx.event.EventHandler;
//import javafx.scene.control.Button;
//import javafx.scene.input.MouseEvent;
//import javafx.scene.text.Font;
//import org.bytedeco.opencv.presets.opencv_core;
//import org.datavec.api.io.labels.ParentPathLabelGenerator;
//import org.datavec.api.split.FileSplit;
//import org.datavec.image.loader.NativeImageLoader;
//import org.datavec.image.recordreader.ImageRecordReader;
//import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
//import org.deeplearning4j.nn.api.Layer;
//import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
//import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
//import org.deeplearning4j.nn.graph.ComputationGraph;
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
//import org.deeplearning4j.util.ModelSerializer;
//import org.nd4j.evaluation.classification.Evaluation;
//import org.nd4j.evaluation.classification.ROC;
//import org.nd4j.evaluation.curves.RocCurve;
//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
//import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
//import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
//import javafx.application.Application;
//import javafx.embed.swing.SwingFXUtils;
//import javafx.geometry.Pos;
//import javafx.scene.Scene;
//import javafx.scene.canvas.Canvas;
//import javafx.scene.control.Label;
//import javafx.scene.image.ImageView;
//import javafx.scene.layout.HBox;
//import javafx.scene.layout.VBox;
//import javafx.stage.Stage;
//import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//import javax.imageio.ImageIO;
//import javax.swing.*;
//
//
///**
// * Test UI for Crop classifier obtained from transfer learning.  <br>
// * Run the {@link org.deeplearning4j.examples.vgg16.FeaturizedPreSave} first to build the model.
// *
// * @author marcial sg
// */
//
//public class CropModelEnsamble extends Application {
//    DataNormalization trainImageScaler;
//    private static final Logger LOGGER = LoggerFactory.getLogger(CropModelEnsamble.class);
//
//    private static final String SURFACE_TERMINAL_BASE_PATH = "E:\\02-17-20 Backup\\Documents\\IIASA";
//    private static final String IIASA_TERMINAL_BASE_PATH = "H:\\MyDocuments";
//    private static final String TITAN_TERMINAL_BASE_PATH = "C:\\Users\\HP WorkStation\\Documents\\Documentos Titan\\IIASA";
//    private static final String TERMINAL_BASE_PATH = TITAN_TERMINAL_BASE_PATH;
//    private static final String BASE_PATH = TERMINAL_BASE_PATH + "\\Image Dataset";
//    private static final String PICTURE_PILE_PATH = TITAN_TERMINAL_BASE_PATH + "\\earthchallengeImages";
//    private static final String PICTURE_PILE_BATCHES_PATH = PICTURE_PILE_PATH + "\\batches";
//
//
//
//    static private String newline = "\n";
//    private static int CANVAS_WIDTH = 1200;
//    private static int CANVAS_HEIGHT = 800;
//    private static int BATCH_SIZE = 100;
//    private static int IMAGE_WIDTH;
//    private static int IMAGE_HEIGHT;
//    private static int IMAGE_CHANNELS;
//    Label lblResult = new Label();
//    Label sampleLabel = new Label();
//    int outputNum = 5; // 2 croptype classification
//    // array of supported extensions (use a List if you prefer)
//    static final String[] EXTENSIONS = new String[]{
//        "gif", "png", "bmp", "jpg"// and other formats you need
//    };
//    JFileChooser fc = null, modelChooser = null;
//    Canvas canvas;
//    ImageView imgView;
//    String newImagePath = null;
//
//    String[] classNames1 = {"maize", "wheat"};
//    String[] classNames2 = {"grape", "sunflower"};
//    String[] classNames3 = {"grape", "maize", "sunflower", "wheat"};
//    String[] classNames4 = {"daisy", "dandelion", "roses", "sunflowers", "tulips"};
//    String[] classNames5 = {"maize", "other", "wheat"};
//    String[] cropModelTrainingPrefix = {"MW", "GS", "MWGS", "FLOWERS", "MWO"};
//    String[] classLabels;
//    String cropModelPrefix;
//    int cropTypeIndex = 4;//  MWO
//
//    int PICTURE_PILE_BATCH_NO = 2;
//
//    String[] cropModelTrainingPaths = {
//        BASE_PATH + "\\" + cropModelTrainingPrefix[0],
//        BASE_PATH + "\\" + cropModelTrainingPrefix[1],
//        BASE_PATH + "\\" + cropModelTrainingPrefix[2],
//        BASE_PATH + "\\" + cropModelTrainingPrefix[3],
//        BASE_PATH + "\\" + cropModelTrainingPrefix[4]};
//
//    ArrayList<String[]> classNames = new ArrayList();
//
//    private final double[] maizeCentroid = {0.965910816,	0.520882138,	0.013386066,	0.977219936};
//    private final double[] otherCentroid = {0.489447644,	0.826917604,	0.625937269,	0.818640766};
//    private final double[] wheatCentroid =  {0.815427272,	0.531788577,	0.939501779,	0.949467153};
//    private final double[][] centroids = {maizeCentroid, otherCentroid ,wheatCentroid};
//
//    final int[] resnetMixMaxValues = {0,998};
//
//    MultiLayerNetwork multiLayerNetwork;
//    ComputationGraph vgg16Model; // base model
//    ComputationGraph resnet50Model; // base model
//
//
//    // filter to identify images based on their extensions
//    static final FilenameFilter IMAGE_FILTER = new FilenameFilter() {
//
//        @Override
//        public boolean accept(final File dir, final String name) {
//
//            if (dir.isDirectory() && !(name.endsWith(".ini")) && !(name.endsWith(".db") && !(name.endsWith(".json")))) {
//                return true;
//            }
//
//            for (final String ext : EXTENSIONS) {
//
//                if (name.endsWith("." + ext)) {
//                    return (true);
//                }
//            }
//            return (false);
//        }
//    };
//
//    DataSetIterator testIter;
//    FileSplit testSplit;
//    ParentPathLabelGenerator labelMaker;
//    private int TLMODEL_IMAGE_HEIGHT;
//    private int TLMODEL_IMAGE_WIDTH;
//    private final int VGG16_CODE = 100;
//    private final int MW_CODE = 200;
//    private final int RESNET50_CODE = 300;
//    private String testResult;
//
//    public CropModelEnsamble() throws IOException {
//    }
//
//    public static void main(String[] args) {
//        launch();
//    }
//
//    private void initSampleInputParams() {
//
//        FeedForwardToCnnPreProcessor preProcessor = (FeedForwardToCnnPreProcessor) multiLayerNetwork.getLayerWiseConfigurations().getInputPreProcessors().get(0);
//        IMAGE_HEIGHT = (int) preProcessor.getInputHeight();
//        IMAGE_WIDTH = (int) preProcessor.getInputWidth();
//
//        String trainDatPath = BASE_PATH + "\\" + cropModelPrefix + "\\trn";
//        File trainData = new File(trainDatPath);
//        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // use parent directory name as the image label
//        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, new Random(1234));
//
//        ImageRecordReader trainRR = new ImageRecordReader(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, labelMaker);
//
//        try {
//            trainRR.initialize(trainSplit);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, BATCH_SIZE, 1, outputNum);
//
//        // pixel values from 0-255 to 0-1 (min-max scaling)
//        DataNormalization imageScaler = new ImagePreProcessingScaler();
//        imageScaler.fit(trainIter);
//
//        trainIter.setPreProcessor(imageScaler);
//
//        // pixel values from 0-255 to 0-1 (min-max scaling)
//        trainImageScaler = new ImagePreProcessingScaler();
//        trainImageScaler.fit(trainIter);
//
//    }
//
//    private Evaluation showEvalStats() {
//        // vectorization of test data
//
//        labelMaker = new ParentPathLabelGenerator();
//
//        File testData = new File(BASE_PATH + "\\" + cropModelPrefix + "\\test");
//        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, new Random());
//        ImageRecordReader testRR = new ImageRecordReader(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, labelMaker);
//
//        try {
//            testRR.initialize(testSplit);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//        testIter = new RecordReaderDataSetIterator(testRR, BATCH_SIZE, 1, outputNum);
//        testIter.setPreProcessor(new VGG16ImagePreProcessor()); // same ImageRecordReader normalization for better results
//        Evaluation eval = resnet50Model.evaluate(testIter);
//        LOGGER.info(eval.stats());
//        return eval;
//
//    }
//
//    private void setModel(String modelPath) throws IOException {
//
//        File model = new File(modelPath);
//
//        if (!model.exists()) {
//            throw new IOException("Can't find the model");
//        } else {
//            LOGGER.info("model file exists");
//        }
//
//        multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(model);
//
//        initSampleInputParams();
//        // Evaluation evaluation = showEvalStats();
//        //showROCStats(evaluation);
//
//    }
//
//    private void setPreTrainedModelModel(String modelPath, int modelCode) throws IOException {
//
//
//        File model = new File(modelPath);
//
//        if (!model.exists()) {
//            throw new IOException("Can't find the model");
//        } else {
//            LOGGER.info("model file exists");
//        }
//
//        switch (modelCode) {
//            case RESNET50_CODE:
//                resnet50Model = ComputationGraph.load(model, true);
//                break;
//            case VGG16_CODE:
//                vgg16Model = ComputationGraph.load(model, true);
//                break;
//        }
//
//        // Evaluation evaluation = showEvalStats();
//        //showROCStats(evaluation);
//
//    }
//
//    private void showROCStats(Evaluation eval) {
//
//        //class 1
//        //ROC STATS
//        ROC roc = resnet50Model.evaluateROC(testIter);
//        LOGGER.info("ROC stats: " + roc.stats());
//        roc.getRocCurve();
//
//        RocCurve rocCurve = roc.getRocCurve();
//        double[] y = rocCurve.getY();
//        double[] x = rocCurve.getX();
//        double[] threshold = rocCurve.getThreshold();
//
//        LOGGER.info(eval.confusionMatrix());
//        int cn = eval.classCount(1);
//        int cp = eval.classCount(0);
//
//        Map<Integer, Integer> tpMap = eval.truePositives();
//
//        int counter = 0;
//        int[] tp = new int[2];
//        tp[0] = 0;
//        tp[1] = 0;
//
//        while (counter < tpMap.size()) {
//            LOGGER.info("value " + counter + " : " + tpMap.get(counter));
//
//            tp[counter] = tpMap.get(counter);
//
//            counter++;
//        }
//
//        LOGGER.info("CLASS : " + classLabels[0]);
//        LOGGER.info("SPECIFICITY = " + ((double) tp[0] / cn));
//        LOGGER.info("SENSITIVITY = " + ((double) tp[1] / cp));
//        LOGGER.info("MCC: " + eval.scoreForMetric(Evaluation.Metric.MCC));
//        LOGGER.info("PRECISION: " + eval.scoreForMetric(Evaluation.Metric.PRECISION));
//        LOGGER.info("RECALL:" + eval.scoreForMetric(Evaluation.Metric.RECALL));
//        LOGGER.info("F1:" + eval.scoreForMetric(Evaluation.Metric.F1));
//
//
//    }
//
//    @Override
//    public void start(Stage stage) {
//
//
//        classNames.add(classNames1);
//        classNames.add(classNames2);
//        classNames.add(classNames3);
//        classNames.add(classNames4);
//        classNames.add(classNames5);
//
//        classLabels = classNames.get(cropTypeIndex);
//        cropModelPrefix = cropModelTrainingPrefix[cropTypeIndex];
//
//        LOGGER.info("");
//        canvas = new Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
//
//        imgView = new ImageView();
//        imgView.setFitHeight(200);
//        imgView.setFitWidth(300);
//
//
//        lblResult.setWrapText(true);
//        lblResult.setFont(new Font("Arial", 12));
//
//        sampleLabel.setWrapText(true);
//        sampleLabel.setFont(new Font("Arial", 12));
//
//        IMAGE_CHANNELS = 3;
//        TLMODEL_IMAGE_HEIGHT = 224;
//        TLMODEL_IMAGE_WIDTH = 224;
//
//        String vgg16ModelPath = TERMINAL_BASE_PATH + "\\Transfer Learning models\\crops\\VGG16.zip";
//        String resnet50ModelPath = TERMINAL_BASE_PATH + "\\Transfer Learning models\\crops\\resnet50.zip";
//        String mwModelPath = TERMINAL_BASE_PATH + "\\Transfer Learning models\\crops\\10222019235106_RELU_MSE_E14_model.zip";
//
//        try {
//            setPreTrainedModelModel(resnet50ModelPath, RESNET50_CODE);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//        try {
//            setModel(mwModelPath);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//        for(int i = 10 ; i < 70 ; i++){
//
//            createPicturePileCSV(i);
//        }
//
////        System.out.println(resnet50Model.summary());
////
////
////        newImagePath = BASE_PATH + "\\GS\\validation\\sunflower\\sunflower_lucas2 (3).jpg";
////        updateImagePreview(newImagePath);
////
////        Button createCSVButton = new Button("Create CSV");
////        createCSVButton.setOnMouseClicked(new EventHandler<MouseEvent>() {
////            @Override
////            public void handle(MouseEvent event) {
////                LOGGER.info("createCSVButton BUTTON CLICKED");
////                createPicturePileCSV();
////
////            }
////        });
////
////        Button selectImageButton = new Button("Select Image");
////        selectImageButton.setOnMouseClicked(new EventHandler<MouseEvent>() {
////            @Override
////            public void handle(MouseEvent event) {
////                LOGGER.info("selectImageButton BUTTON CLICKED");
////
////                lblResult.setText("");
////                newImagePath = chooseTestImage();
////                LOGGER.info("newImagePath: " + newImagePath);
////
////                updateImagePreview(newImagePath);
////
////            }
////        });
////
////
////        Button testAllImagesButton = new Button("Test All");
////        testAllImagesButton.setOnMouseClicked(new EventHandler<MouseEvent>() {
////            @Override
////            public void handle(MouseEvent event) {
////                LOGGER.info("testAllImagesButton BUTTON CLICKED");
////                testAllImages();
////
////            }
////        });
////
////
////
////        Button testButton = new Button("Test Image");
////        testButton.setOnMouseClicked(new EventHandler<MouseEvent>() {
////            @Override
////            public void handle(MouseEvent event) {
////                LOGGER.info("Test BUTTON CLICKED");
////
////                String[] output1 = testImage(newImagePath, RESNET50_CODE);
////                String[] output2 = testImage(newImagePath, MW_CODE);
////
////                double normalizedLabelOutput = normalizeResnetOutput1(output1[0]);
////                double[] sampleDoubles = {normalizedLabelOutput,	Double.parseDouble(output1[1]),	Double.parseDouble(output2[0]),	Double.parseDouble(output2[1])};
////
////                testResult = classifySample(sampleDoubles);
////                LOGGER.info("testResult : " + testResult);
////
////                lblResult.setText(testResult);
////
////            }
////        });
////
////        HBox imgViewBox1 = new HBox(5);
////        imgViewBox1.getChildren().addAll(imgView, sampleLabel);
////        imgViewBox1.setAlignment(Pos.CENTER);
////
////        HBox buttonBox1 = new HBox(5);
////        buttonBox1.getChildren().addAll(testAllImagesButton,createCSVButton);
////        buttonBox1.setAlignment(Pos.CENTER);
////
////        HBox buttonBox2 = new HBox(5);
////        buttonBox2.getChildren().addAll(selectImageButton, testButton);
////        buttonBox2.setAlignment(Pos.CENTER);
////
////        VBox root = new VBox(10);
////        root.getChildren().addAll(imgViewBox1, lblResult, buttonBox2, buttonBox1);
////        root.setAlignment(Pos.CENTER);
////
////        Scene scene = new Scene(root, 800, 500);
////        stage.setScene(scene);
////        stage.setTitle("Transfer Learning Model Test");
////        stage.setResizable(false);
////        stage.show();
////
////        canvas.requestFocus();
//    }
//
//    private String classifySample(double[] sampleDoubles) {
//
//        String predictedLabel = "";
//
//        int[] featureIndex = {0,1,3};
//
//        double[] selectedSampleFeatures = new double[featureIndex.length];
//        double[][] classCentroidsSelectedFeatures = new double[centroids.length][featureIndex.length];
//
//        for(int i = 0 ; i < featureIndex.length  ; i++){
//
//            selectedSampleFeatures[i]  = sampleDoubles[featureIndex[i]];
//
//            classCentroidsSelectedFeatures[0][i] = centroids[0][featureIndex[i]];
//            classCentroidsSelectedFeatures[1][i] = centroids[1][featureIndex[i]];
//            classCentroidsSelectedFeatures[2][i] = centroids[2][featureIndex[i]];
//
//        }
//
//        double[] distances = new double[centroids.length];
//
//        for (int i = 0 ; i < centroids.length ; i++){
//
//            distances[i] = getEuclideanDistance(selectedSampleFeatures,   classCentroidsSelectedFeatures[i]);
//        }
//
//        int minDistanceIndex = getMinDistance(distances);
//
//        predictedLabel = classLabels[minDistanceIndex];
//
//        return predictedLabel;
//    }
//
//    private int getMinDistance(double[] distances) {
//
//        double minValue = 1000;
//        int minIndex = -1;
//
//        for(int i = 0 ; i < distances.length ; i++){
//
//            if(distances[i] < minValue){
//                minValue = distances[i];
//                minIndex = i ;
//            }
//
//        }
//        return minIndex;
//    }
//
//    private double getEuclideanDistance(double[] selectedSampleFeatures, double[] classCentroidsSelectedFeature) {
//
//        double sum = 0;
//
//        for(int i = 0 ; i < selectedSampleFeatures.length ; i++){
//
//            sum +=  Math.pow(selectedSampleFeatures[i] - classCentroidsSelectedFeature[i],2);
//
//        }
//
//        double result = Math.sqrt(sum);
//
//        return result;
//    }
//
//    /**
//     * This method classifies the input image.
//     *
//     * @param  newImagePath Filepath with the test image to be classified.
//     * @return String[] class label on index 0, and confidence value on index 1.
//     */
//    private String[] testImage(String newImagePath, int modelCode) {
//
//        String[] result = new String[2];
//        String prediction = "", probability = "";
//
//        int predictionClass;
//        double probabilityValue;
//
//        switch (modelCode) {
//
//            case RESNET50_CODE:
//                try {
//                    NativeImageLoader loader = new NativeImageLoader(TLMODEL_IMAGE_HEIGHT, TLMODEL_IMAGE_WIDTH, IMAGE_CHANNELS);
//                    INDArray image = loader.asMatrix(new File(newImagePath));
//
//                    VGG16ImagePreProcessor imagePreProcessor = new VGG16ImagePreProcessor();
//                    imagePreProcessor.transform(image);
//
//                    INDArray[] samples = {image};
//                    INDArray[] outputPrediction = resnet50Model.output(false, samples);
//
//                    predictionClass = (int) outputPrediction[0].argMax(1).getDouble(0);
//                    probabilityValue = outputPrediction[0].getDouble(predictionClass);
//
//                    prediction = String.valueOf(predictionClass);
//                    probability = String.valueOf(probabilityValue);
//
//                } catch (IOException e) {
//                    e.printStackTrace();
//                }
//
//                break;
//
//            default:
//
//                NativeImageLoader loader = new NativeImageLoader(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS);
//
//                try {
//
//                    INDArray image = loader.asMatrix(new File(newImagePath));
//                    trainImageScaler.transform(image);
//
//                    int[] outputPrediction = multiLayerNetwork.predict(image);
//                    prediction = String.valueOf(outputPrediction[0]);
//
//                    INDArray output = multiLayerNetwork.output(image);
//                    probability = String.valueOf(output.getDouble(outputPrediction[0]));
//
//                } catch (IOException e) {
//                    e.printStackTrace();
//                }
//
//                break;
//        }
//
//
//        result[0] = prediction;
//        result[1] = probability;
//
//        return result;
//    }
//
//    private void createPicturePileCSV(int batch) {
//
//        String imagesPath = PICTURE_PILE_BATCHES_PATH + "\\batch" + batch;
//
//        File dir = new File(imagesPath);
//
//        String outputDataset = "";
//        String titles = "image_name" + "," + "label" + "," + "tf_model_label" + "," + "tf_model_value" + "," + "mw_model_label" + "," + "mw_model_value" + "\n";
//        outputDataset += titles;
//
//        for (final File imgFile : dir.listFiles(IMAGE_FILTER)) {
//
//            BufferedImage img = null;
//            String imgLabel = imgFile.getName().split("_")[1];
//            System.out.println("imgLabel: " + imgLabel);
//
//            try {
//
//                img = ImageIO.read(imgFile);
//                LOGGER.info("image : " + imgFile.getName());
//
//                if (img != null) {
//
//                    String[] output1 = testImage(imgFile.getPath(), RESNET50_CODE);
//                    String[] output2 = testImage(imgFile.getPath(), MW_CODE);
//
//                    DecimalFormat df = new DecimalFormat("0.000000"); //TODO: CHECK IF 6 FLOATING POINTS ARE ENOUGH
//
//                    String sample = imgFile.getName() + "," + imgLabel + "," + output1[0] + "," + df.format(Double.parseDouble(output1[1])) + "," + output2[0] + "," + df.format(Double.parseDouble(output2[1])) + "\n";
//                    outputDataset += sample;
//
//                }
//
//            } catch (final IOException e) {
//                // handle errors here
//            }
//
//        }
//
//        saveDatasetToCSV(outputDataset, batch);
//
//        LOGGER.info("CSV File Created");
//    }
//
//
//    private void saveDatasetToCSV(String outputDataset) {
//
//        String pattern = "MMddyyyyHHmmss";
//        SimpleDateFormat simpleDateFormat = new SimpleDateFormat(pattern);
//        String date = simpleDateFormat.format(new Date());
//        String csvPath = TERMINAL_BASE_PATH + "\\cropModelEnsambleVal_"+ date +".csv";
//
//        try {
//            FileWriter fw = new FileWriter(csvPath,true);
//            BufferedWriter bw = new BufferedWriter(fw);
//            bw.write(outputDataset);
//            bw.flush();
//
//        } catch (IOException e) {
//            System.out.println("FileWriter IOException: " + e.toString());
//        }
//
//
//    }
//
//    private void saveDatasetToCSV(String outputDataset, int batch) {
//
//        String pattern = "MMddyyyyHHmmss";
//        SimpleDateFormat simpleDateFormat = new SimpleDateFormat(pattern);
//        String date = simpleDateFormat.format(new Date());
//        String csvPath = TERMINAL_BASE_PATH + "\\cropModelEnsambleVal_"+ date +"_batch"+ batch+".csv";
//
//        try {
//            FileWriter fw = new FileWriter(csvPath,true);
//            BufferedWriter bw = new BufferedWriter(fw);
//            bw.write(outputDataset);
//            bw.flush();
//
//        } catch (IOException e) {
//            System.out.println("FileWriter IOException: " + e.toString());
//        }
//
//
//    }
//
//    private void updateImagePreview(String newImagePath) {
//        try {
//            BufferedImage myPicture = ImageIO.read(new File(newImagePath));
//            imgView.setImage(SwingFXUtils.toFXImage(myPicture, null));
//        } catch (IOException e) {
//            LOGGER.error(e.toString());
//        }
//
//
//    }
//
//    private String chooseModel() {
//        String result;
//        String fileChooserDefaultPath = TERMINAL_BASE_PATH + "\\best models";
//        File defaultDirectory = new File(fileChooserDefaultPath);
//
//        //Set up the file chooser.
//        if (modelChooser == null) {
//
//            modelChooser = new JFileChooser();
//            modelChooser.setSelectedFile(defaultDirectory);
//        }
//
//        modelChooser.setPreferredSize(new Dimension(1000, 1000));
//
//        //Show it.
//        int returnVal = modelChooser.showDialog(null,
//            "Select model");
//
//        //Process the results.
//        if (returnVal == JFileChooser.APPROVE_OPTION) {
//            File file = modelChooser.getSelectedFile();
//            result = file.getPath();
//
//
//            modelChooser.setSelectedFile(new File(result));
//        } else {
//            LOGGER.info("Selection cancelled by user." + newline);
//            result = null;
//            //Reset the file chooser for the next time it's shown.
//            modelChooser.setSelectedFile(defaultDirectory);
//        }
//
//        return result;
//
//    }
//
//    private String chooseTestImage() {
//        String result;
//        String fileChooserDefaultPath = TERMINAL_BASE_PATH + "\\Image Dataset\\.*";
//        File defaultDirectory = new File(fileChooserDefaultPath);
//
//        //Set up the file chooser.
//        if (fc == null) {
//
//            fc = new JFileChooser();
//            fc.setSelectedFile(defaultDirectory);
//        }
//
//        fc.setPreferredSize(new Dimension(1000, 1000));
//        fc.setFileFilter(new ImageFilter());
//        //Show it.
//        int returnVal = fc.showDialog(null,
//            "Select image");
//
//        //Process the results.
//        if (returnVal == JFileChooser.APPROVE_OPTION) {
//            File file = fc.getSelectedFile();
//            result = file.getPath();
//            sampleLabel.setText(file.getName());
//            fc.setSelectedFile(new File(result));
//        } else {
//            LOGGER.info("Selection cancelled by user." + newline);
//            result = null;
//            //Reset the file chooser for the next time it's shown.
//            fc.setSelectedFile(defaultDirectory);
//        }
//
//        return result;
//
//
//    }
//
//    private void testAllImages() {
//
//        String imagesPath = "C:\\Users\\HP WorkStation\\Desktop\\validation";
//        File dir = new File(imagesPath);
//
//        int truePositives;
//        int classSamples;
//
//        for (final File f : dir.listFiles(IMAGE_FILTER)) {
//
//            System.out.println("file label: " + f.getName());
//
//
//            for (File imgFile : f.listFiles(IMAGE_FILTER)) {
//
//                classSamples = f.listFiles(IMAGE_FILTER).length;
//
////                BufferedImage img = null;
////
////                try {
////
////                    img = ImageIO.read(imgFile);
////                    LOGGER.info("image : " + imgFile.getName());
////                    LOGGER.info("class: " + f.getName());
////
////                    if (img != null) {
////
////                        String[] output1 = testImage(newImagePath, RESNET50_CODE);
////                        String[] output2 = testImage(newImagePath, MW_CODE);
////
////                        double normalizedLabelOutput = normalizeResnetOutput1(output1[0]);
////                        double[] sampleDoubles = {normalizedLabelOutput,	Double.parseDouble(output1[1]),	Double.parseDouble(output2[0]),	Double.parseDouble(output2[1])};
////
////                        String output = classifySample(sampleDoubles);
////                        LOGGER.info("output : " + output);
////
////                        if( f.getName().equals(output)){
////                            truePositives++;
////                        }
////
////                    }
////
////                    double classAcc = ((double) truePositives)/((double) classSamples);
////                    LOGGER.info("classAcc : " + classAcc);
////
////                } catch (final IOException e) {
////                    // handle errors here
////                }
//            }
//
//        }
//
//    }
//
//    private void createCSV() {
//
////        String imagesPath = cropModelTrainingPaths[cropTypeIndex] + "\\train";
//        //String imagesPath = "C:\\Users\\HP WorkStation\\Desktop\\validation";
//        String imagesPath = PICTURE_PILE_BATCHES_PATH + "\\batch" + PICTURE_PILE_BATCH_NO;
//
//        File dir = new File(imagesPath);
//
//        String outputDataset = "";
//        String titles = "image_name" + "," + "label" + "," + "tf_model_label" + "," + "tf_model_value" + "," + "mw_model_label" + "," + "mw_model_value" + "\n";
//        outputDataset += titles;
//
//        int sampleLabelIndex = -1;
//        for (final File f : dir.listFiles(IMAGE_FILTER)) {
//            sampleLabelIndex++;
//            LOGGER.info("class: " + f.getName());
//
////            for (int i = 0 ; i < 10 ; i++) { //TEST CODE
////
////                File[] listFiles = f.listFiles(IMAGE_FILTER);
////                File imgFile = listFiles[i];
//
//            for (File imgFile : f.listFiles(IMAGE_FILTER)) {
//
//                BufferedImage img = null;
//
//                try {
//
//                    img = ImageIO.read(imgFile);
//                    LOGGER.info("image : " + imgFile.getName());
//
//                    if (img != null) {
//
//                        String[] output1 = testImage(imgFile.getPath(), RESNET50_CODE);
//                        String[] output2 = testImage(imgFile.getPath(), MW_CODE);
//
//                        DecimalFormat df = new DecimalFormat("0.000000"); //TODO: CHECK IF 6 FLOATING POINTS ARE ENOUGH
//
//                        String sample = imgFile.getName() + "," + sampleLabelIndex + "," + output1[0] + "," + df.format(Double.parseDouble(output1[1])) + "," + output2[0] + "," + df.format(Double.parseDouble(output2[1])) + "\n";
//                        outputDataset += sample;
//
//                    }
//
//                } catch (final IOException e) {
//                    // handle errors here
//                }
//            }
//
//        }
//
//        saveDatasetToCSV(outputDataset);
//
//        LOGGER.info("CSV File Created");
//    }
//
//    private double normalizeResnetOutput1(String outputValue) {
//        double result = -1;
//
//        double value = Double.parseDouble(outputValue);
//        result = (value - resnetMixMaxValues[0]) / ( resnetMixMaxValues[1] - resnetMixMaxValues[0]);
//
//        return result;
//    }
//
//}
