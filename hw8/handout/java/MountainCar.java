import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

public class MountainCar {

  private double minPosition;
  private double maxPosition;
  private double maxSpeed;
  private double goalPosition;
  private double force;
  private double gravity;

  private double [] low;
  private double [] high;
  private double [] state;

  private int actionSpace;

  public int getActionSpace() {
    return actionSpace;
  }

  public int getStateSpace() {
    return stateSpace;
  }

  private int stateSpace;

  private Random r;
  private String mode;
  private IHT iht;

  public MountainCar(String mode) {
    this.minPosition = -1.2;
    this.maxPosition = 0.6;
    this.maxSpeed = 0.07;
    this.goalPosition = 0.5;
    this.force = 0.001;
    this.gravity = 0.0025;
    this.actionSpace = 3;

    low = new double[2];
    high = new double[2];
    state = new double[2];

    if (mode.equals("tile"))
      stateSpace = 2048;
    else if (mode.equals("raw"))
      stateSpace = 2;
    else
      throw new IllegalArgumentException("Invalid environment mode. Must be tile or raw");
    this.mode = mode;

    r = new Random();

    seed(0);
    reset();
  }

  public HashMap<Integer, Double> transform(double[] state) {
    double position = state[0];
    double velocity = state[1];

    position = (position + 1.2) / 1.8;
    velocity = (velocity + 0.07) / 0.14;

    assert (0 <= position && position <= 1);
    assert (0 <= velocity && position <= 1);

    position *= 2;
    velocity *= 2;

    if (mode.equals("tile")) {
      if (iht == null) {
        iht = new IHT(stateSpace);
      }

      ArrayList<Integer> tiling = IHT.tiles(iht, 64, new double[] {position, velocity}, new int[] {0}, false);
      tiling.addAll(IHT.tiles(iht, 64, new double[] {position}, new int[] {1}, false));
      tiling.addAll(IHT.tiles(iht, 64, new double[] {velocity}, new int[] {2}, false));

      HashMap<Integer, Double> returnMap = new HashMap<>();
      for (int index : tiling) {
        returnMap.put(index, 1.0);
      }
      return returnMap;
    } else if (mode.equals("raw")) {
      HashMap<Integer, Double> returnMap = new HashMap<>();
      returnMap.put(0, state[0]);
      returnMap.put(1, state[1]);
      return returnMap;
    } else {
      throw new IllegalArgumentException("Invalid environment mode. Must be tile or raw");
    }
  }

  public void seed(int seed) {
    r.setSeed(seed);
  }

  HashMap<Integer, Double> reset() {
    state[0] = -0.5; //nextRandom();
    state[1] = 0;
    return transform(state);
  }

  public NextState step(int action) {
    assert (action == 0 || action == 1 || action == 2);

    double position = state[0];
    double velocity = state[1];

    velocity += (action - 1) * force + Math.cos(3 * position) * -gravity;
    velocity = clip(velocity, -maxSpeed, maxSpeed);
    position += velocity;
    position = clip(position, minPosition, maxPosition);

    if (position == minPosition && velocity < 0)
      velocity = 0;

    boolean done = position >= goalPosition;
    double reward = -1.0;

    state[0] = position;
    state[1] = velocity;

    return new NextState(transform(state), reward, done);
  }

  private double nextRandom() {
    double rand = r.nextDouble();
    return -0.6 + (0.2) * rand;
  }

  static double clip(double a, double min, double max) {
    if (a > max)
      return max;
    else if (a < min)
      return min;
    return a;
  }

  public static class NextState {
    private HashMap<Integer, Double> states;
    private double reward;
    private boolean done;

    public NextState(HashMap<Integer, Double> states, double reward, boolean done) {
      this.states = states;
      this.reward = reward;
      this.done = done;
    }

    public HashMap<Integer, Double> getStates() {
      return states;
    }

    public double getReward() {
      return reward;
    }

    public boolean isDone() {
      return done;
    }
  }
}