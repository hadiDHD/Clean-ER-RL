package exp;

import model.*;
import kotlin.Pair;
import model.Module;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;

public class ExpState implements Encodable {

    public static final int MAX_ELEMENTS = 35;

    static final int INPUT_SIZE = MAX_ELEMENTS * (MAX_ELEMENTS + 1) / 2;

    static final int OUTPUT_SIZE = MAX_ELEMENTS * (MAX_ELEMENTS - 1) / 2;

    public ArrayList<Module> modules;

    byte[] input;

    ERModel er;

    double reward = 0;

    private ExpState() {
        super();
    }

    public ExpState(ERModel er) {
        super();
        this.er = er;
        input = new byte[INPUT_SIZE];

        modules = new ArrayList<>(MAX_ELEMENTS);
        for (int i = 0; i < er.elements.size(); i++) {
            Module m = new Module();
            Element element = er.elements.get(i);
            m.addElement(element);
            modules.add(i, m);
        }
//        for (int i = er.elements.size(); i < MAX_ELEMENTS; i++) {
//            modules.add(i, new Module());
//        }
        reward = ExpMQ.apply(this);
    }

    public void resetInput() {
        Arrays.fill(input, (byte) 0);
    }

    public void decreaseInput(int a, int b) {
        input[getTriangularMatrixIndex(a, b)]--;
    }

    public void increaseInput(int a) {
        input[getTriangularMatrixIndex(a, a)]++;
    }

    public Module moveItoJ(int i, int j) {
        Module mI = modules.get(i);
        Module mJ = modules.get(j);
        mJ.merge(mI);
        modules.remove(mI);
        return mJ;
    }

    public static int getTriangularMatrixIndex(int a, int b) {
        // https://stackoverflow.com/a/27682124
        if (a < b) {
            int temp = b;
            b = a;
            a = temp;
        }
        int size = a * (a + 1) / 2;
        return size + b;
    }

    public static Pair<Integer, Integer> getTriangularMatrixRowAndColumn(int index) {
        // https://math.stackexchange.com/a/1417583
        double numerator = Math.sqrt(8 * index + 1) + 1;
        int row = (int) Math.floor(numerator / 2) - 1;
        int column = index - row * (row + 1) / 2;
        if (column > row || column < 0 || row < 0) {
            throw new RuntimeException("WRONG!");
        }
        return new Pair<Integer, Integer>(row, column);
    }

    @Override
    public Encodable dup() {
        ExpState newState = new ExpState();
        newState.er = er;
        newState.input = Arrays.copyOf(input, input.length);
        newState.modules = deepCopyModules();
        newState.reward = reward;
        return newState;
    }

    private ArrayList<Module> deepCopyModules() {
        ArrayList<Module> newSet = new ArrayList<>(modules.size());
        for (int i = 0; i < modules.size(); i++) {
            newSet.add(i, modules.get(i).copy());
        }
        return newSet;
    }

    @Override
    public INDArray getData() {
        return Nd4j.create(input, new long[]{input.length}, DataType.INT8);
    }

    @Override
    public boolean isSkipped() {
        return false;
    }

    @Override
    public double[] toArray() {
        return getData().toDoubleVector();
    }

    public void print() {
        int i = 0;
        for (Module m : modules) {
            if (m.isEmpty()) {
                continue;
            }
            i++;
            System.out.println(m.toString(i));
        }
        System.out.println("MQ Index = " + getMQ());
        System.out.println("Reward = " + getReward());
    }

//    public double getReward() {
//        double sum = 0.0;
//        double diff = 0.0;
//        for (Module m : modules) {
//            sum += m.getMQ().getMQ() * m.size();
//            diff += Math.abs(m.size() - 7);
//        }
//        return 12.0 * sum / er.elements.size() - diff / modules.size();
//    }

    public double getReward() {
        return getMQ() / modules.size() - getSD7() / 18;
    }

    public double getMQ() {
        double sum = 0.0;
        for (Module m : modules) {
            sum += m.getMQ().getMQ();
        }
        return sum;
    }

    public double getSD7() {
        double sum = 0;
        for (Module m : modules) {
            sum += Math.pow(m.size() - 7, 2);
        }
        return Math.sqrt(sum / modules.size());
    }

}
