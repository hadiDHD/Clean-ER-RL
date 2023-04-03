package exp;

import model.*;
import com.google.gson.Gson;
import model.Module;
import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.configuration.A3CLearningConfiguration;
import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.network.configuration.ActorCriticDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.nd4j.linalg.learning.config.Adam;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class ExpTrainerStagedA3C {

    static int epochStep = ExpState.MAX_ELEMENTS - 1;

    private static final ActorCriticDenseNetworkConfiguration NET_A3C = ActorCriticDenseNetworkConfiguration.builder()
            .numLayers(3)
            .numHiddenNodes(ExpState.OUTPUT_SIZE)
            .learningRate(0.01)
            .updater(new Adam(0.001))
            .l2(0.000)
            .build();

    public static void main(String[] args) {
        try {
            System.setProperty("org.bytedeco.openblas.load", "mkl");
            // create Gson instance
            Gson gson = new Gson();

            JFileChooser chooser = new JFileChooser();

            File directory = new File("./dataset");
            chooser.setCurrentDirectory(directory);
            chooser.showOpenDialog(null);

            // create a reader
            Reader reader = Files.newBufferedReader(Paths.get(chooser.getSelectedFile().getAbsolutePath()));

            // convert JSON string to User object
            ERModel model = gson.fromJson(reader, ERModel.class);

            model.init();
            System.out.println(model.elements.size());

            // close reader
            reader.close();
            ExpMDP mdp = new ExpMDP();
            MyPolicy policy = null;
            //should save the in-between NNs
            boolean shouldSaveInBetweenNNs = true;
            MyA3CDiscreteDense dql;
            final List<ERModel> split = HCS.split(model);
            for (int i = 0; i < 1; i++) {
                A3CLearningConfiguration LEARNING_A3C =
                        A3CLearningConfiguration.builder()
                                .maxEpochStep(epochStep)
                                .maxStep(5000000)
                                .numThreads(64)
                                .rewardFactor(0.01)
                                .gamma(0.99)
                                .nStep(5)
                                .build();
                if (policy == null) {
                    dql = new MyA3CDiscreteDense(mdp, NET_A3C, LEARNING_A3C);
                } else {
                    dql = new MyA3CDiscreteDense(mdp, policy.getNeuralNet(), LEARNING_A3C);
                }
                dql.addListener(new TrainingListener() {
                    @Override
                    public ListenerResponse onTrainingStart() {
//                        System.out.println("onTrainingStart");
                        return ListenerResponse.CONTINUE;
                    }

                    @Override
                    public void onTrainingEnd() {
//                        System.out.println("onTrainingStart");
                    }

                    @Override
                    public ListenerResponse onNewEpoch(IEpochTrainer trainer) {
//                        System.out.println("onTrainingStart");
                        return ListenerResponse.CONTINUE;
                    }

                    @Override
                    public ListenerResponse onEpochTrainingResult(IEpochTrainer trainer, IDataManager.StatEntry statEntry) {
//                        System.out.println("onEpochTrainingResult");
                        return ListenerResponse.CONTINUE;
                    }

                    @Override
                    public ListenerResponse onTrainingProgress(ILearning learning) {
                        System.out.println("onTrainingProgress");
                        System.out.println("stepCount:" + learning.getStepCount());
                        MyPolicy policy = ((MyPolicy) learning.getPolicy()).copy();
                        new Thread(() -> {
                            List<Module> modules = new ArrayList<>();
                            for (ERModel m : split) {
                                ExpState state = play(policy, m);
                                modules.addAll(state.modules);
                            }
                            System.out.println("Overall MQ: " + getMQ(modules, model));
                            System.out.println("Standard Deviation from seven: " + getStandardDeviationFromSeven(modules));
                            printModules(modules);
                            //should save or not

                            if (shouldSaveInBetweenNNs) {
                                try {
                                    String name = System.currentTimeMillis() + "";
                                    File valueN = new File("neural networks/between/" + name + "_value");
                                    File policyN = new File("neural networks/between/" + name + "_policy");
                                    if (!valueN.exists()) {
                                        valueN.createNewFile();
                                    }
                                    if (!policyN.exists()) {
                                        policyN.createNewFile();
                                    }
                                    System.out.println(valueN.getAbsolutePath());
                                    policy.save(valueN.getAbsolutePath(), policyN.getAbsolutePath());
                                } catch (IOException e) {
                                    e.printStackTrace();
                                }
                            }
                        }).start();
                        return ListenerResponse.CONTINUE;
                    }
                });
                dql.train();

                try {
                    String name = System.currentTimeMillis() + "";
                    File valueN = new File("neural networks/" + name + "_value");
                    File policyN = new File("neural networks/" + name + "_policy");
                    if (!valueN.exists()) {
                        valueN.createNewFile();
                    }
                    if (!policyN.exists()) {
                        policyN.createNewFile();
                    }
                    System.out.println(valueN.getAbsolutePath());
                    policy.save(valueN.getAbsolutePath(), policyN.getAbsolutePath());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            List<Module> modules = new ArrayList<>();
            for (ERModel m : split) {
                ExpState state = play(policy, m);
                modules.addAll(state.modules);
            }
            System.out.println("Overall MQ: " + getMQ(modules, model));
            System.out.println("Standard Deviation from seven: " + getStandardDeviationFromSeven(modules));
            printModules(modules);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static ExpState play(MyPolicy policy, ERModel model) {
        ExpMDP playMdp = new ExpMDP(model);
        ExpState bestState = policy.play(playMdp);
        for (int p = 0; p < 100; p++) {
            playMdp = new ExpMDP(model);
            ExpState newState = policy.play(playMdp);
            if (newState.reward > bestState.reward) {
                bestState = newState;
            }
        }
        return bestState;
    }

    public static double getMQ(List<Module> modules, ERModel er) {
        for (Module m : modules) {
            m.getMQ().reset();
        }
        for (Relationship r : er.relationships) {
            int moduleR = -1;
            int moduleA = -1;
            int moduleB = -1;
            for (int i = 0; i < modules.size(); i++) {
                Module m = modules.get(i);
                if (m.contains(r)) {
                    moduleR = i;
                }
                if (m.contains(r.a)) {
                    moduleA = i;
                }
                if (m.contains(r.b)) {
                    moduleB = i;
                }
            }
            if (moduleR == -1) {
                continue;
            }
            if (moduleA != -1) {
                if (moduleA == moduleR) {
                    modules.get(moduleA).getMQ().intraLink++;
                } else {
                    modules.get(moduleA).getMQ().interLink++;
                    modules.get(moduleR).getMQ().interLink++;
                }
            }
            if (moduleB != -1) {
                if (moduleB == moduleR) {
                    modules.get(moduleB).getMQ().intraLink++;
                } else {
                    modules.get(moduleB).getMQ().interLink++;
                    modules.get(moduleR).getMQ().interLink++;
                }
            }
        }
        for (Generalization g : er.generalizations) {
            int moduleParent = -1;
            int moduleChild = -1;
            for (int i = 0; i < modules.size(); i++) {
                Module m = modules.get(i);
                if (m.contains(g.parent)) {
                    moduleParent = i;
                }
                if (m.contains(g.child)) {
                    moduleChild = i;
                }
                if (moduleChild == -1 || moduleParent == -1) {
                    continue;
                }
                if (moduleParent == moduleChild) {
                    modules.get(moduleParent).getMQ().intraLink++;
                } else {
                    modules.get(moduleParent).getMQ().interLink++;
                    modules.get(moduleChild).getMQ().interLink++;
                }
            }
        }
        double sum = 0;
        for (Module m : modules) {
            sum += m.getMQ().getMQ();
        }
        return sum;
    }

    public static double getStandardDeviationFromSeven(List<Module> modules) {
        double sum = 0;
        for (Module m : modules) {
            sum += Math.pow(Math.abs(m.getElementsArray().length - 7), 2);
        }
        return Math.sqrt(sum / modules.size());
    }

    public static void printModules(List<Module> modules) {
        for (int i = 0; i < modules.size(); i++) {
            System.out.println(modules.get(i).toString(i + 1));
        }
    }


}

