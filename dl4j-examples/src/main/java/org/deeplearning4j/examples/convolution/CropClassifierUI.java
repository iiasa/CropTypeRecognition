///*******************************************************************************
// * Copyright (c) 2015-2019 Skymind, Inc.
// *
// * This program and the accompanying materials are made available under the
// * terms of the Apache License, Version 2.0 which is available at
// * https://www.apache.org/licenses/LICENSE-2.0.
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// * License for the specific language governing permissions and limitations
// * under the License.
// *
// * SPDX-License-Identifier: Apache-2.0
// ******************************************************************************/
//
//package org.deeplearning4j.examples.convolution;
//
//import java.awt.*;
//import java.awt.image.BufferedImage;
//import java.io.*;
//import java.text.DecimalFormat;
//import java.util.ArrayList;
//import java.util.Map;
//import java.util.Random;
//
////import javafx.event.EventHandler;
////import javafx.scene.control.Button;
////import javafx.scene.input.MouseEvent;
////import javafx.scene.text.Font;
//import org.datavec.api.io.labels.ParentPathLabelGenerator;
//import org.datavec.api.split.FileSplit;
//import org.datavec.image.loader.NativeImageLoader;
//import org.datavec.image.recordreader.ImageRecordReader;
//import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
////import javafx.application.Application;
////import javafx.embed.swing.SwingFXUtils;
////import javafx.geometry.Pos;
////import javafx.scene.Scene;
////import javafx.scene.canvas.Canvas;
////import javafx.scene.control.Label;
////import javafx.scene.image.ImageView;
////import javafx.scene.layout.HBox;
////import javafx.scene.layout.VBox;
////import javafx.stage.Stage;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//import javax.imageio.ImageIO;
//import javax.swing.*;
//import javax.ws.rs.core.Application;
//
//
///**
// * Test UI for MNIST classifier. User can painting digit by using mouse and predict value using a trained model. <br>
// * Run the {@link CropClassifier} first to build the model.
// *
// * @author jesuino
// * @author fvaleri
// * @author dariuszzbyrad
// */
//public class CropClassifierUI extends Application {
//    DataNormalization trainImageScaler;
//    private static final Logger LOGGER = LoggerFactory.getLogger(CropClassifierUI.class);
//
//    private static final String SURFACE_TERMINAL_BASE_PATH = "E:\\02-17-20 Backup\\Documents\\IIASA";
//    private static final String IIASA_TERMINAL_BASE_PATH = "H:\\MyDocuments";
//    private static final String TITAN_TERMINAL_BASE_PATH = "C:\\Users\\HP WorkStation\\Documents\\IIASA";
//    private static final String TERMINAL_BASE_PATH = TITAN_TERMINAL_BASE_PATH;
//    private static final String BASE_PATH = TERMINAL_BASE_PATH + "\\Image Dataset";
//
//    static private String newline = "\n";
//    private static int CANVAS_WIDTH = 150;
//    private static int CANVAS_HEIGHT = 150;
//    private static int BATCH_SIZE = 100;
//    private static int IMAGE_WIDTH;
//    private static int IMAGE_HEIGHT;
//    private static int IMAGE_CHANNELS;
//    Label lblResult = new Label();
//    Label sampleLabel = new Label();
//    int outputNum = 5; // 2 croptype classification
//    // array of supported extensions (use a List if you prefer)
//    static final String[] EXTENSIONS = new String[]{
//            "gif", "png", "bmp", "jpg"// and other formats you need
//    };
//    JFileChooser fc = null, modelChooser = null;
//    Canvas canvas;
////    ImageView imgView;
//    String newImagePath = null;
//
//    String[] classNames1 = {"maize", "wheat"};
//    String[] classNames2 = {"grape", "sunflower"};
//    String[] classNames3 = {"grape", "maize", "sunflower", "wheat"};
//    String[] classNames4 = {"daisy","dandelion","roses","sunflowers","tulips"};
//    String[] cropModelTrainingPrefix = {"MW", "GS", "MWGS", "FLOWERS"};
//    String[] classLabels;
//    String cropModelPrefix;
//
//    String[] cropModelTrainingPaths = {
//            BASE_PATH + "\\" + cropModelTrainingPrefix[0],
//            BASE_PATH + "\\" + cropModelTrainingPrefix[1],
//            BASE_PATH + "\\" + cropModelTrainingPrefix[2],
//            BASE_PATH + "\\" + cropModelTrainingPrefix[3]};
//
//    ArrayList<String[]> classNames = new ArrayList();
//
//
//    private MultiLayerNetwork net; // trained model
//    // filter to identify images based on their extensions
//    static final FilenameFilter IMAGE_FILTER = new FilenameFilter() {
//
//        @Override
//        public boolean accept(final File dir, final String name) {
//
//            if (dir.isDirectory() && !(name.endsWith(".ini")) && !(name.endsWith(".db"))) {
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
//    DataSetIterator testIter;
//    FileSplit testSplit;
//    ParentPathLabelGenerator labelMaker;
//
//    public CropClassifierUI() throws IOException {
//
//
//    }
//
//
//
//    public static void main(String[] args) {
//        launch();
//    }
//
//    private void initSampleInputParams() {
//        FeedForwardToCnnPreProcessor preProcessor = (FeedForwardToCnnPreProcessor) net.getLayerWiseConfigurations().getInputPreProcessors().get(0);
//        IMAGE_HEIGHT = (int) preProcessor.getInputHeight();
//        IMAGE_WIDTH = (int) preProcessor.getInputWidth();
//        IMAGE_CHANNELS = (int) preProcessor.getNumChannels();
//
//        LOGGER.info("inputHeight : " + IMAGE_HEIGHT);
//        LOGGER.info("inputWidth : " + IMAGE_WIDTH);
//        LOGGER.info("inputChannels : " + IMAGE_CHANNELS);
//
//        String trainDatPath = BASE_PATH + "\\"+cropModelPrefix+"\\train";
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
//        File testData = new File(BASE_PATH + "\\"+cropModelPrefix+ "\\test");
//        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS,  new Random());
//        ImageRecordReader testRR = new ImageRecordReader(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, labelMaker);
//
//        try {
//            testRR.initialize(testSplit);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//        testIter = new RecordReaderDataSetIterator(testRR, BATCH_SIZE, 1, outputNum);
//        testIter.setPreProcessor(trainImageScaler); // same ImageRecordReader normalization for better results
//        Evaluation eval = net.evaluate(testIter);
//        LOGGER.info(eval.stats());
//        return eval;
//
//    }
//
//    private void setModel(String modelPath) throws IOException {
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
//        net = ModelSerializer.restoreMultiLayerNetwork(model);
//
//
//        initSampleInputParams();
//       // Evaluation evaluation = showEvalStats();
//        //showROCStats(evaluation);
//
//    }
//
//    private void showROCStats(Evaluation eval) {
//
//        //class 1
//        //ROC STATS
//        ROC roc = net.evaluateROC(testIter);
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
//        int counter = 0 ;
//        int[] tp = new int[2];
//        tp[0] = 0 ;
//        tp[1] = 0 ;
//
//        while (counter < tpMap.size()){
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
//
//    }
//    // fi
//
//    @Override
//    public void start(Stage stage) {
//
//        int cropTypeIndex = 3;//  FLOWERS
//
//        classNames.add(classNames1);
//        classNames.add(classNames2);
//        classNames.add(classNames3);
//        classNames.add(classNames4);
//
//        classLabels = classNames.get(cropTypeIndex);
//        cropModelPrefix = cropModelTrainingPrefix[cropTypeIndex];
//
//        LOGGER.info("");
//        canvas = new Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
//
//        imgView = new ImageView();
//        imgView.setFitHeight(56);
//        imgView.setFitWidth(100);
//
//
//        lblResult.setWrapText(true);
//        lblResult.setFont(new Font("Arial", 10));
//
//        sampleLabel.setWrapText(true);
//        sampleLabel.setFont(new Font("Arial", 8));
//
////        String modelLabel = "10242019213957_IDENTITY_MSE_E14_model";  //GS MODEL
////        String defaultModelPath = TERMINAL_BASE_PATH + "\\Selected Models\\" +modelLabel+".zip";
//        String defaultModelPath = "C:\\Users\\HP WorkStation\\Documents\\IIASA\\Transfer Learning models\\12092020220457.zip";
//
//        LOGGER.info("defaultModelPath: " + defaultModelPath);
//
//
//
//        newImagePath = BASE_PATH + "\\GS\\validation\\sunflower\\sunflower_lucas2 (3).jpg";
//        LOGGER.info("newImagePath : " + newImagePath);
//
//        updateImagePreview(newImagePath);
//
//        try {
//            setModel(defaultModelPath);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
////        ModelImageConvolution modelImageConvolution = new ModelImageConvolution(net,defaultTestImage);
////
////        double [][][][] convolutedImage = null;
////
////        convolutedImage = modelImageConvolution.getConvolutedArray();
//
//
//
//        Button selectModelButton = new Button("Select Model");
//        selectModelButton.setOnMouseClicked(new EventHandler<MouseEvent>() {
//            @Override
//            public void handle(MouseEvent event) {
//                LOGGER.info("selectModel BUTTON CLICKED");
//
//                String slectedModelPath = chooseModel();
//
//                LOGGER.info("newModel Path : " + slectedModelPath);
//
//                try {
//                    setModel(slectedModelPath);
//                } catch (IOException e) {
//                    LOGGER.info("selectModel setModel ERROR : " + e.toString());
//                }
//
//            }
//        });
//
//        Button testAllImagesButton = new Button("Test All");
//        testAllImagesButton.setOnMouseClicked(new EventHandler<MouseEvent>() {
//            @Override
//            public void handle(MouseEvent event) {
//                LOGGER.info("testAllImagesButton BUTTON CLICKED");
//                testAllImages();
//
//            }
//        });
//
//        Button selectImageButton = new Button("Select Image");
//        selectImageButton.setOnMouseClicked(new EventHandler<MouseEvent>() {
//            @Override
//            public void handle(MouseEvent event) {
//                LOGGER.info("selectImageButton BUTTON CLICKED");
//
//                lblResult.setText("");
//                newImagePath = chooseTestImage();
//                LOGGER.info("newImagePath: " + newImagePath);
//
//                updateImagePreview(newImagePath);
//
//            }
//        });
//
//        Button testButton = new Button("Test Image");
//        testButton.setOnMouseClicked(new EventHandler<MouseEvent>() {
//            @Override
//            public void handle(MouseEvent event) {
//                LOGGER.info("Test BUTTON CLICKED");
//
//                String[] output = testImage(newImagePath);
//
//
//                String labelText = String.format("Prediction: %s \nProbability: %s", output[0], output[1]);
//
//                lblResult.setText(labelText);
//
//            }
//        });
//
//        HBox imgViewBox1 = new HBox(5);
//        imgViewBox1.getChildren().addAll(imgView, sampleLabel);
//        imgViewBox1.setAlignment(Pos.CENTER);
//
//        HBox buttonBox1 = new HBox(5);
//        buttonBox1.getChildren().addAll(selectModelButton, testAllImagesButton);
//        buttonBox1.setAlignment(Pos.CENTER);
//
//        HBox buttonBox2 = new HBox(5);
//        buttonBox2.getChildren().addAll(selectImageButton, testButton);
//        buttonBox2.setAlignment(Pos.CENTER);
//
//        VBox root = new VBox(20);
//        root.getChildren().addAll(imgViewBox1, lblResult, buttonBox1, buttonBox2);
//        root.setAlignment(Pos.CENTER);
//
//        Scene scene = new Scene(root, 300, 300);
//        stage.setScene(scene);
//        stage.setTitle("Wheat/Maize Model Test");
//        stage.setResizable(false);
//        stage.show();
//
//        canvas.requestFocus();
//    }
//
//    private String[] testImage(String newImagePath) {
//
//        String[] result = new String[2];
//        String prediction = "", probability = "";
//
//        NativeImageLoader loader = new NativeImageLoader(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS);
//
//        try {
//
//            INDArray image = loader.asMatrix(new File(newImagePath));
//            trainImageScaler.transform(image);
//
//            int[] outputPrediction = net.predict(image);
//            prediction = classLabels[outputPrediction[0]];
//
//            INDArray output = net.output(image);
//            probability = String.valueOf(output.getDouble(outputPrediction[0]) * 100);
//
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//        result[0] = prediction;
//        result[1] = probability;
//        LOGGER.info("");
//
//        return result;
//    }
//
//    private void testAllImages() {
//
//        String imagesPath = TERMINAL_BASE_PATH + "\\Image Dataset\\"+cropModelPrefix+"\\validation";
//        File dir = new File(imagesPath);
//
//        //LOGGER.info("imagesInPath = " +  dir.listFiles(IMAGE_FILTER).length);
//        int truePositives = 0;
//        int testSamples = 0;
//
//        ArrayList<String> falseNegativesNames = new ArrayList<>();
//
//        for (final File f : dir.listFiles(IMAGE_FILTER)) {
//
//            LOGGER.info("class: " + f.getName());
//
//            for (File imgFile : f.listFiles(IMAGE_FILTER)) {
//
//                BufferedImage img = null;
//
//                try {
//
//                    img = ImageIO.read(imgFile);
//
//                    if (img != null) {
//
//                        // you probably want something more involved here
//                        // to display in your U
//
//                        String[] output = testImage(imgFile.getPath());
//                        String ouputClassLabel = output[0];
//
//                        testSamples += 1;
//                        LOGGER.info(testSamples + "-" + imgFile.getName() + "  - " + ouputClassLabel);
//
//                        if (ouputClassLabel.equals(f.getName())) {
//                            truePositives++;
//                        } else {
//
//                            falseNegativesNames.add(imgFile.getName());
//                        }
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
//        LOGGER.info("testSamples: " + testSamples);
//        LOGGER.info("truePositives: " + truePositives);
//
//        DecimalFormat df = new DecimalFormat("0.0000");
//        LOGGER.info("Accuracy: " + df.format((float) truePositives / (float) testSamples));
//
//        LOGGER.info("");
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
//                "Select model");
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
//                "Select image");
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
//    private BufferedImage getScaledImage(String imgPath) {
//
//        BufferedImage bf = null;
//        try {
//            bf = ImageIO.read(new File(imgPath));
//        } catch (IOException ex) {
//            LOGGER.error("Image failed to load.");
//        }
//
////        LOGGER.info("original width: " + bf.getWidth());
////        LOGGER.info("original height: " + bf.getHeight());
//
//        Image tmp = bf.getScaledInstance(IMAGE_WIDTH, IMAGE_HEIGHT, Image.SCALE_SMOOTH);
//        BufferedImage bufferedImage = new BufferedImage(IMAGE_WIDTH, IMAGE_HEIGHT,
//                BufferedImage.TYPE_BYTE_INDEXED);
//
//        //  Image.SCALE_REPLICATE , BufferedImage.TYPE_INT_RGB : Accuracy: 0.4950
//        //  Image.SCALE_REPLICATE , BufferedImage.SCALE_DEFAULT : Accuracy: 0.4950
//        //  Image.SCALE_SMOOTH , BufferedImage.BITMASK :Accuracy: 0.4950
//        //  Image.SCALE_SMOOTH , BufferedImage.TYPE_INT_ARGB_PRE :Accuracy: 0.4950
//        //  Image.SCALE_REPLICATE , BufferedImage.TYPE_INT_ARGB_PRE :Accuracy: 0.4950
//        //  Image.SCALE_SMOOTH , BufferedImage.TYPE_INT_ARGB :Accuracy: 0.4950
//        //  Image.SCALE_SMOOTH , BufferedImage.TYPE_3BYTE_BGR : Accuracy: 0.6139
//        //  Image.SCALE_SMOOTH , BufferedImage.SCALE_SMOOTH : Accuracy: 0.6535
//        //  Image.SCALE_SMOOTH , BufferedImage.TYPE_INT_BGR : Accuracy: 0.6535
//        //  Image.SCALE_SMOOTH , BufferedImage.TYPE_USHORT_555_RGB : Accuracy: 0.6634
//        //  Image.SCALE_REPLICATE , BufferedImage.SCALE_SMOOTH :Accuracy: 0.8020
//        //  Image.SCALE_REPLICATE , BufferedImage.TYPE_INT_BGR :Accuracy: 0.8020
//        //  Image.SCALE_SMOOTH , BufferedImage.TYPE_BYTE_GRAY : Accuracy =  0.81
//
//
//        //  Image.SCALE_REPLICATE , BufferedImage.TYPE_BYTE_INDEXED :  Accuracy: 0.8713
//        //  Image.SCALE_DEFAULT , BufferedImage.TYPE_BYTE_INDEXED : Accuracy: 0.8713
//        //  Image.SCALE_AREA_AVERAGING , BufferedImage.TYPE_BYTE_INDEXED : Accuracy: 0.8812
//        //  Image.SCALE_SMOOTH , BufferedImage.TYPE_BYTE_INDEXED : Accuracy =  0.8812
//
//        Graphics g = bufferedImage.createGraphics();
//        g.drawImage(tmp, 0, 0, null);
//        g.dispose();
//
//        return bufferedImage;
//
//    }
//
//    private String[] predictImage(BufferedImage img) throws IOException {
//
//        NativeImageLoader loader = new NativeImageLoader(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS);
//        INDArray imageINDArray = loader.asMatrix(img);
//        trainImageScaler.transform(imageINDArray);
//        INDArray output = net.output(imageINDArray);
//        int predictedDigit = net.predict(imageINDArray)[0];
//        double probability = output.getDouble(predictedDigit) * 100;
//        String[] result = {String.valueOf(predictedDigit), String.valueOf(probability)};
//
//        return result;
//    }
//
//}
