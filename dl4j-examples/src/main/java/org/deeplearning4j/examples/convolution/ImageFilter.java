package org.deeplearning4j.examples.convolution;

import org.deeplearning4j.examples.utilities.DataUtilities;

import java.io.File;
import javax.swing.*;
import javax.swing.filechooser.*;

/* ImageFilter.java is used by FileChooserDemo2.java. */
public class ImageFilter extends FileFilter {

    //Accept all directories and all gif, jpg, tiff, or png files.
    public boolean accept(File f) {
        if (f.isDirectory()) {
            return true;
        }

        String extension = DataUtilities.getExtension(f);
        if (extension != null) {
            if (extension.equals(DataUtilities.tiff) ||
                    extension.equals(DataUtilities.tif) ||
                    extension.equals(DataUtilities.gif) ||
                    extension.equals(DataUtilities.jpeg) ||
                    extension.equals(DataUtilities.jpg) ||
                    extension.equals(DataUtilities.png)) {
                return true;
            } else {
                return false;
            }
        }

        return false;
    }

    //The description of this filter
    public String getDescription() {
        return "Images";
    }
}
