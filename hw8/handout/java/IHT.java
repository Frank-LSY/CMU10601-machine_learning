import java.util.ArrayList;
import java.util.HashMap;

public class IHT <T> {
  private int size;
  private int overfullCount;
  HashMap<T, Integer> dictionary;

  public IHT(int sizeval) {
    size = sizeval;
    overfullCount = 0;
    dictionary = new HashMap<>();
  }

  int count() {
    return dictionary.size();
  }

  boolean fullp() {
    return dictionary.size() >= size;
  }

  int getindex(T obj, boolean readonly) {
    if (dictionary.containsKey(obj))
      return dictionary.get(obj);
    else if (readonly)
      return -1;

    if (count() >= size) {
      if (overfullCount == 0)
      {
        System.out.println("IHT full, starting to allow collisions");
      }
      overfullCount++;
      return obj.hashCode() % size;
    } else {
      int count = count();
      dictionary.put(obj, count);
      return count;
    }

  }

  public String toString() {
    return "Collision table: size: " + size + " overfullCount: " + overfullCount + " dictionary: " + dictionary.size()
            + "items";
  }

  public static <T> int hashcoords (T coordinates, IHT<T> m, boolean readonly) {
    return m.getindex(coordinates, readonly);
  }

  public static <T> ArrayList<Integer> tiles(IHT iht, int i, double[] floats, int[] ints, boolean readonly) {
    ArrayList<Integer> qfloats = new ArrayList<>();
    for (double f: floats) {
      qfloats.add((int) Math.floor(f * i));
    }
    ArrayList<Integer> tiles = new ArrayList<>();
    for(int tiling = 0; tiling < i; tiling++) {
      int tilingX2 = tiling *2;
      ArrayList<Integer> coords = new ArrayList<>(tiling);
      coords.add(tiling);
      int b = tiling;

      for (int q: qfloats) {
        coords.add((q + b) / i);
        b += tilingX2;
      }

      for (int x: ints) {
        coords.add(x);
      }
      tiles.add(hashcoords(coords, iht, readonly));
    }
    return tiles;
  }

}
