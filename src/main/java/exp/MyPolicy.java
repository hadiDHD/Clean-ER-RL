package exp;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.network.ac.ActorCriticSeparate;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.LegacyMDPWrapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public class MyPolicy extends ACPolicy<ExpState> {

    private IActorCritic actorCritic;
    Random rnd;

    public MyPolicy(IActorCritic actorCritic) {
        super(actorCritic);
        this.actorCritic = actorCritic;
    }

    public MyPolicy(IActorCritic actorCritic, Random rnd) {
        super(actorCritic, rnd);
        this.actorCritic = actorCritic;
        this.rnd = rnd;
    }

    public static MyPolicy load(String pathValue, String pathPolicy) throws IOException {
        Random r = Nd4j.getRandom();
        r.setSeed(System.currentTimeMillis());
        return new MyPolicy(ActorCriticSeparate.load(pathValue, pathPolicy), r);
    }

    public MyPolicy copy() {
        Random r = Nd4j.getRandom();
        r.setSeed(System.currentTimeMillis());
        return new MyPolicy(actorCritic.clone(), r);
    }

    public Integer nextAction(Observation obs, Integer validActionRange) {
        return nextAction(obs.getData(), validActionRange);
    }

    private Integer nextAction(INDArray input, Integer validActionRange) {
        INDArray output = actorCritic.outputAll(input)[1];
        if (rnd == null) {
            return Learning.getMaxAction(output);
        }
        float rVal = rnd.nextFloat();
        double wastedSum = 0;
        for (int i = validActionRange; i < output.length(); i++) {
            wastedSum += output.getFloat(i);
        }
        float excess = (float) (wastedSum / validActionRange);
        for (int i = 0; i < validActionRange; i++) {
            //System.out.println(i + " " + rVal + " " + output.getFloat(i));
            if (rVal < (output.getFloat(i) + excess)) {
                return i;
            } else
                rVal -= (output.getFloat(i) + excess);
        }
        System.out.println("validActionRange: " + validActionRange);
        throw new RuntimeException("Output from network is not a probability distribution: " + output);
    }

    public ExpState play(ExpMDP mdp) {
        resetNetworks();

        LegacyMDPWrapper<ExpState, Integer, DiscreteSpace> mdpWrapper = new LegacyMDPWrapper<>(mdp, null);

        Learning.InitMdp<Observation> initMdp = refacInitMdp(mdpWrapper, null);
        Observation obs = initMdp.getLastObs();

        double reward = initMdp.getReward();

        int lastAction = mdpWrapper.getActionSpace().noOp();
        int action;

        ExpState bestState = (ExpState) mdp.state.dup();

        while (!mdpWrapper.isDone()) {

            int validRange = mdp.getValidActionRange();

            if (obs.isSkipped()) {
                action = lastAction;
            } else {
                action = nextAction(obs, validRange);
            }

            lastAction = action;

            StepReply<Observation> stepReply = mdpWrapper.step(action);
            if (mdp.state.reward > bestState.reward) {
                bestState = (ExpState) mdp.state.dup();
            }
            reward += stepReply.getReward();

            obs = stepReply.getObservation();
        }

        return bestState;
    }
}
