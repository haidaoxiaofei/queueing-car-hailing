/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package riderdispatcher.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 *
 * @author bigstone
 */
public class OutputWriter {
    
    FileWriter fw;
    public OutputWriter(String outputFilePath) throws IOException{
        File f = new File(outputFilePath.substring(0, outputFilePath.lastIndexOf("/")));
        if (!f.exists()) {
           f.mkdirs();
        }
        
        fw = new FileWriter(outputFilePath);
    }
    public void output(String outputString) throws IOException{
        fw.write(outputString);
    }
    
    public void close() throws IOException{
        fw.flush();
        fw.close();
    }
}
