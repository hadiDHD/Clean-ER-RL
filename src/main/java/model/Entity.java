package model;

public class Entity extends Element{
    public Entity(String name) {
        super(name);
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof Entity && super.equals(o);
    }
}
