package exp;

/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 * Copyright (c) 2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/


import org.deeplearning4j.rl4j.learning.async.AsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.learning.async.IAsyncGlobal;
import org.deeplearning4j.rl4j.learning.configuration.A3CLearningConfiguration;
import org.deeplearning4j.rl4j.learning.configuration.IAsyncLearningConfiguration;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/23/16.
 * Training for A3C in the Discrete Domain
 * <p>
 * All methods are fully implemented as described in the
 * https://arxiv.org/abs/1602.01783 paper.
 */
public class MyA3CDiscrete extends AsyncLearning<ExpState, Integer, DiscreteSpace, IActorCritic> {

    final public A3CLearningConfiguration configuration;
    final protected MDP<ExpState, Integer, DiscreteSpace> mdp;
    final private IActorCritic iActorCritic;
    final private AsyncGlobal asyncGlobal;
    final private ACPolicy<ExpState> policy;

    public MyA3CDiscrete(MDP<ExpState, Integer, DiscreteSpace> mdp, IActorCritic iActorCritic, A3CLearningConfiguration conf) {
        this.iActorCritic = iActorCritic;
        this.mdp = mdp;
        this.configuration = conf;
        asyncGlobal = new AsyncGlobal<>(iActorCritic, conf);

        Long seed = conf.getSeed();
        Random rnd = Nd4j.getRandom();
        if (seed != null) {
            rnd.setSeed(seed);
        }

        policy = new MyPolicy(iActorCritic, rnd);
    }

    @Override
    public IPolicy<Integer> getPolicy() {
        return policy;
    }

    @Override
    public IAsyncLearningConfiguration getConfiguration() {
        return configuration;
    }

    @Override
    public MDP<ExpState, Integer, DiscreteSpace> getMdp() {
        return mdp;
    }

    @Override
    protected AsyncThread newThread(int i, int deviceAffinity) {
        return new MyA3CThreadDiscrete(mdp.newInstance(), asyncGlobal, (A3CLearningConfiguration) this.getConfiguration(), deviceAffinity, getListeners(), i);
    }

    @Override
    protected IAsyncGlobal<IActorCritic> getAsyncGlobal() {
        return asyncGlobal;
    }

    public IActorCritic getNeuralNet() {
        return iActorCritic;
    }

    public static class A3CConfiguration {

        int seed;
        int maxEpochStep;
        int maxStep;
        int numThread;
        int nstep;
        int updateStart;
        double rewardFactor;
        double gamma;
        double errorClamp;

        /**
         * Converts the deprecated A3CConfiguration to the new LearningConfiguration format
         */
        public A3CLearningConfiguration toLearningConfiguration() {
            return A3CLearningConfiguration.builder()
                    .seed(new Long(seed))
                    .maxEpochStep(maxEpochStep)
                    .maxStep(maxStep)
                    .numThreads(numThread)
                    .nStep(nstep)
                    .rewardFactor(rewardFactor)
                    .gamma(gamma)
                    .build();

        }

    }
}
