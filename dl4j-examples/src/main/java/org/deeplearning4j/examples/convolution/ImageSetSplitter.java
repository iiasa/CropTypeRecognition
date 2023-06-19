package org.deeplearning4j.examples.convolution;

import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.Random;

public class ImageSetSplitter {

    private static String imagePath = "C:\\Users\\HP WorkStation\\Documents\\IIASA\\Image Dataset\\FLOWERS";

    // array of supported extensions (use a List if you prefer)
    static final String[] EXTENSIONS = new String[]{
        "gif", "png", "bmp", "jpg"// and other formats you need
    };

    // filter to identify images based on their extensions
    static final FilenameFilter IMAGE_FILTER = new FilenameFilter() {

        @Override
        public boolean accept(final File dir, final String name) {

            if (dir.isDirectory() && !(name.endsWith(".ini")) && !(name.endsWith(".db")) && !(name.endsWith(".txt"))) {
                if (!(name.equals("train"))) {
                    return true;
                }

            }

            for (final String ext : EXTENSIONS) {

                if (name.endsWith("." + ext)) {
                    return true;
                }
            }

            return false;
        }
    };

    public static void main(String... args) {

        Path temp = null;

        File dir = new File(imagePath);

        String trainDirLabel = "trn";
        String testDirLabel = "test";
        String valDirLabel = "val";
        String[] targetFolders = {trainDirLabel, testDirLabel, valDirLabel};

        double trainProportion = 0.8;
        double testProportion = (1 - trainProportion) / 2;
        double valProportion = (1 - trainProportion) / 2;

        File[] classDirectories = dir.listFiles(IMAGE_FILTER);

        for (final File classDir : classDirectories) {

            System.out.println("CLASS: " + classDir.getName() + "\n\n");

            ArrayList<ArrayList<Integer>> trainTestValIndexes = getTrainTestValIndexes(trainProportion, testProportion, valProportion, classDir);

            File[] classFiles = classDir.listFiles(IMAGE_FILTER);

            for (int i = 0 ; i < classFiles.length ; i++ ) {

                File imgFile = classFiles[i];

                for (int j = 0 ; j < targetFolders.length ; j++) {

                    String targetFolder = targetFolders[j];
                    ArrayList<Integer> setIndexes = trainTestValIndexes.get(j);
                //    System.out.println(targetFolder + " Quantity: " + setIndexes.size());

                    if(setIndexes.contains(i)){

                   //     System.out.println("image: " + imgFile.getName());

                        String beginPath = imagePath + "\\" + classDir.getName() + "\\" + imgFile.getName();

                        File targetFolderDirectory = checkCreateFile(imagePath + "\\" + targetFolder);
                        File classFolderDirectory = checkCreateFile(targetFolderDirectory.getPath()+ "\\" + classDir.getName() );

                        String endPath = classFolderDirectory.getPath() + "\\" + imgFile.getName();

                //        System.out.println("FROM : " + beginPath);
                //        System.out.println("TO : " + endPath);

                        try {
                            temp = Files.move(Paths.get(beginPath), Paths.get(endPath));
                        } catch (IOException e) {
                            System.out.println("ERROR MOVING FILE: " + e.toString());
                        }

                        if (temp != null) {
                       //     System.out.println("File moved successfully");
                        } else {
                        //    System.out.println("Failed to move the file");
                        }

                    }

                }

            }

        }

    }

    private static ArrayList<ArrayList<Integer>> getTrainTestValIndexes(double trainProportion, double testProportion, double valProportion, File classDir) {

        ArrayList<ArrayList<Integer>> trainTestValIndexes = new ArrayList<>();

        int trainSamplesQty = (int) Math.floor(classDir.list().length * trainProportion);
        int testSamplesQty = (int) Math.floor(classDir.list().length * testProportion);
        int valSamplesQty = (int) Math.floor(classDir.list().length * valProportion);

        Random rand = new Random();
        ArrayList<Integer> indexes = new ArrayList<>();

        for (int i = 0; i < classDir.listFiles(IMAGE_FILTER).length; i++) {
            indexes.add(i);
        }


        ArrayList<Integer> trainIndexes = new ArrayList<>();

        for (int i = 0; i < trainSamplesQty; i++) {

            rand = new Random();
            int randomIndex = rand.nextInt(indexes.size());

            trainIndexes.add(indexes.get(randomIndex));

            indexes.remove(randomIndex);
        }

        trainTestValIndexes.add(trainIndexes);


        ArrayList<Integer> testIndexes = new ArrayList<>();

        for (int i = 0; i < testSamplesQty; i++) {
            rand = new Random();
            int randomIndex = rand.nextInt(indexes.size());

            testIndexes.add(indexes.get(randomIndex));

            indexes.remove(randomIndex);
        }


        trainTestValIndexes.add(testIndexes);

        ArrayList<Integer> valIndexes = new ArrayList<>();

        for (int i = 0; i < valSamplesQty; i++) {
            rand = new Random();
            int randomIndex = rand.nextInt(indexes.size());

            valIndexes.add(indexes.get(randomIndex));

            indexes.remove(randomIndex);
        }

        trainTestValIndexes.add(valIndexes);

        return  trainTestValIndexes;
    }

    @NotNull
    private static File checkCreateFile(String filePath) {

       // System.out.println("checkCreateFile : " + filePath);

        File localFile = new File(filePath);

        if (!localFile.exists()) {
         //   System.out.println("File dont exist");
            localFile.mkdir();
        } else {
           // System.out.println("File  exist");
        }

        return localFile;

    }

}
