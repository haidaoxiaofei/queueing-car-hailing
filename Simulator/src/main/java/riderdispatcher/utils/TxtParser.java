package riderdispatcher.utils;

import riderdispatcher.core.SavableObject;

import java.io.*;
import java.util.ArrayList;
import java.util.List;


/**
 * Created by bigstone on 13/6/2017.
 */
public class TxtParser {

    public static List readFromFile(Class<? extends SavableObject> objectClass, String filePath) throws IllegalAccessException, InstantiationException, FileNotFoundException, UnsupportedEncodingException {
        List objects = new ArrayList();
        final SavableObject record = objectClass.newInstance();

        BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(filePath), "utf-8"));

        r.lines().forEach(line ->{
            line = line.trim();
            Object object = record.fromString(line);
            if (object!=null){
                objects.add(object);
            }
        });

        try {
            r.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return objects;
    }




    public static void writeToFile(List<? extends SavableObject> recordObjects, String filePath) throws IOException {
        PrintWriter pw = new PrintWriter(new OutputStreamWriter(new FileOutputStream(filePath), "utf-8"));
        for (SavableObject recordObject : recordObjects) {
            pw.print(recordObject.convertToString());
            pw.print("\n");
        }
        pw.close();
    }




}
