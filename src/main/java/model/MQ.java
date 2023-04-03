package model;


public class MQ {
    // inside link
    public int intraLink;
    // outside link
    public int interLink;

    public MQ() {
        intraLink = 0;
        interLink = 0;
    }

    public void reset() {
        intraLink = 0;
        interLink = 0;
    }

    public double getMQ() {
        if (intraLink > 0) {
            return intraLink / (intraLink + interLink / 2.0);
        }
        return 0;
    }

    public MQ copy() {
        MQ newMQ = new MQ();
        newMQ.intraLink = intraLink;
        newMQ.interLink = interLink;
        return newMQ;
    }

}
