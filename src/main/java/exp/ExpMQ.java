package exp;

import model.*;
import model.Module;

public class ExpMQ {

    public static double apply(ExpState state) {
        for (Module m : state.modules) {
            m.getMQ().reset();
        }
        state.resetInput();
        for (Relationship r : state.er.relationships) {
            int moduleR = -1;
            int moduleA = -1;
            int moduleB = -1;
            for (int i = 0; i < state.modules.size(); i++) {
                Module m = state.modules.get(i);
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
                    state.modules.get(moduleA).getMQ().intraLink++;
                    state.increaseInput(moduleA);
                } else {
                    state.modules.get(moduleA).getMQ().interLink++;
                    state.modules.get(moduleR).getMQ().interLink++;
                    state.decreaseInput(moduleA, moduleR);
                }
            }
            if (moduleB != -1) {
                if (moduleB == moduleR) {
                    state.modules.get(moduleB).getMQ().intraLink++;
                    state.increaseInput(moduleB);
                } else {
                    state.modules.get(moduleB).getMQ().interLink++;
                    state.modules.get(moduleR).getMQ().interLink++;
                    state.decreaseInput(moduleB, moduleR);
                }
            }
        }
        for (Generalization g : state.er.generalizations) {
            int moduleParent = -1;
            int moduleChild = -1;
            for (int i = 0; i < state.modules.size(); i++) {
                Module m = state.modules.get(i);
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
                    state.modules.get(moduleParent).getMQ().intraLink++;
                    state.increaseInput(moduleParent);
                } else {
                    state.modules.get(moduleParent).getMQ().interLink++;
                    state.modules.get(moduleChild).getMQ().interLink++;
                    state.decreaseInput(moduleParent, moduleChild);
                }
            }
        }
        return state.getReward();
    }
}