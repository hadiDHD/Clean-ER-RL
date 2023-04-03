package model;

import java.util.Objects;

public class Generalization {

    public Entity parent;
    public Entity child;

    public Generalization(Entity parent, Entity child) {
        this.parent = parent;
        this.child = child;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Generalization)) return false;
        if (!super.equals(o)) return false;
        Generalization that = (Generalization) o;
        return Objects.equals(parent, that.parent) && Objects.equals(child, that.child);
    }

    @Override
    public int hashCode() {
        return Objects.hash(parent, child);
    }

    @Override
    public String toString() {
        return "Generalization{" +
                "parent=" + parent +
                ", child=" + child +
                '}';
    }
}
