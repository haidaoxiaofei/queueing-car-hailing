package riderdispatcher.core;


import riderdispatcher.utils.TimeUtil;

import java.text.ParseException;

public class OrderRecord implements SavableObject {
    public long pickTime;
    public long dropTime;
    public int pickZoneID;
    public int dropZoneID;
    public int passengerCount;
    public double tripDistance;
    public double totalAmount;

    @Override
    public OrderRecord fromString(String recordString){
        String[] result = recordString.split(",");
        OrderRecord record = new OrderRecord();



        if (result.length <= 1 || result[0].equals("VendorID")) return null;

        //for yellow taxi records
        if (result.length == 17){
            try {
                record.pickTime = TimeUtil.format.parse(result[1]).getTime()/1000;
                record.dropTime = TimeUtil.format.parse(result[2]).getTime()/1000;
            }catch (ParseException e) {
                e.printStackTrace();
            }
            record.pickZoneID = Integer.valueOf(result[7]);
            record.dropZoneID = Integer.valueOf(result[8]);
            record.passengerCount = Integer.valueOf(result[3]);
            record.tripDistance = Double.valueOf(result[4]);
            record.totalAmount = Double.valueOf(result[16]);
        }

        //for green taxi records
        if (result.length == 19){
            try {
                record.pickTime = TimeUtil.format.parse(result[1]).getTime()/1000;
                record.dropTime = TimeUtil.format.parse(result[2]).getTime()/1000;
            }catch (ParseException e) {
                e.printStackTrace();
            }
            record.pickZoneID = Integer.valueOf(result[5]);
            record.dropZoneID = Integer.valueOf(result[6]);
            record.passengerCount = Integer.valueOf(result[7]);
            record.tripDistance = Double.valueOf(result[8]);
            record.totalAmount = Double.valueOf(result[16]);
        }

        //for cleaned taxi records
        if (result.length == 7){
            record.pickTime = Long.valueOf(result[0]);
            record.dropTime = Long.valueOf(result[1]);
            record.pickZoneID = Integer.valueOf(result[2]);
            record.dropZoneID = Integer.valueOf(result[3]);
            record.passengerCount = Integer.valueOf(result[4]);
            record.totalAmount = Double.valueOf(result[5]);
            record.tripDistance = Double.valueOf(result[6]);
        }



        return record;
    }

    @Override
    public String convertToString() {
        StringBuffer orderString = new StringBuffer();
        orderString.append(Long.toString(this.pickTime)).append(",")
                .append(Long.toString(this.dropTime)).append(",")
                .append(this.pickZoneID).append(",")
                .append(this.dropZoneID).append(",")
                .append(this.passengerCount).append(",")
                .append(this.totalAmount).append(",")
                .append(this.tripDistance);
        return orderString.toString();
    }

}
