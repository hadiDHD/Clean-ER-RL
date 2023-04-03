package exp;

import org.deeplearning4j.rl4j.learning.configuration.A3CLearningConfiguration;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraph;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactorySeparate;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactorySeparateStdDense;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.network.configuration.ActorCriticDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.space.DiscreteSpace;

public class MyA3CDiscreteDense extends MyA3CDiscrete {

    public MyA3CDiscreteDense(MDP<ExpState, Integer, DiscreteSpace> mdp, IActorCritic actorCritic, A3CLearningConfiguration conf) {
        super(mdp, actorCritic, conf);
    }

    public MyA3CDiscreteDense(MDP<ExpState, Integer, DiscreteSpace> mdp, ActorCriticFactorySeparate factory,
                              A3CLearningConfiguration conf) {
        this(mdp, factory.buildActorCritic(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf);
    }

    public MyA3CDiscreteDense(MDP<ExpState, Integer, DiscreteSpace> mdp,
                              ActorCriticDenseNetworkConfiguration netConf, A3CLearningConfiguration conf) {
        this(mdp, new ActorCriticFactorySeparateStdDense(netConf), conf);
    }

    public MyA3CDiscreteDense(MDP<ExpState, Integer, DiscreteSpace> mdp, ActorCriticFactoryCompGraph factory,
                              A3CLearningConfiguration conf) {
        this(mdp, factory.buildActorCritic(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf);
    }
}
