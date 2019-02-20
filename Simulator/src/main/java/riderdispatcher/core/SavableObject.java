package riderdispatcher.core;

public interface SavableObject {
    Object fromString(String objectString);
    String convertToString();
}
