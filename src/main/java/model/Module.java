package model;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

public class Module {
    private Set<Entity> entities;
    private Set<Relationship> relationships;
    private MQ MQ = new MQ();

    public Module() {
        entities = new HashSet<>();
        relationships = new HashSet<>();
    }

    public void addElement(Element e) {
        if (e instanceof Entity) {
            entities.add((Entity) e);
        } else if (e instanceof Relationship) {
            relationships.add((Relationship) e);
        }
    }

    public boolean removeElement(Element e) {
        if (e instanceof Entity) {
            return entities.remove((Entity) e);
        } else if (e instanceof Relationship) {
            return relationships.remove((Relationship) e);
        }
        throw new RuntimeException("Element is not in this module!");
//        return false;
    }

    public boolean contains(Element e) {
        if (e instanceof Entity) {
            return entities.contains((Entity) e);
        } else if (e instanceof Relationship) {
            return relationships.contains((Relationship) e);
        }
        return false;
    }

    public MQ getMQ() {
        return MQ;
    }

    public void merge(Module other) {
        entities.addAll(other.entities);
        relationships.addAll(other.relationships);
    }


    public Module copy() {
        Module newModule = new Module();
        newModule.entities = new HashSet<>(entities.size());
        newModule.entities.addAll(entities);
        newModule.relationships = new HashSet<>(relationships.size());
        newModule.relationships.addAll(relationships);
        newModule.MQ = MQ.copy();
        return newModule;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Module)) return false;
        Module module = (Module) o;
        return Objects.equals(entities, module.entities) && Objects.equals(relationships, module.relationships);
    }

    @Override
    public int hashCode() {
        return Objects.hash(entities, relationships);
    }

    @Override
    public String toString() {
        return "Module{" +
                "entities=" + Arrays.toString(entities.toArray()) +
                ", relationships=" + Arrays.toString(relationships.toArray()) +
                ", MQ=" + MQ +
                '}';
    }


    public String toString(int i) {
        return "Module " + i + " {" +
                "entities=" + Arrays.toString(entities.toArray()) +
                ", relationships=" + Arrays.toString(relationships.toArray()) +
                ", MQ=" + MQ.getMQ() +
                '}';
    }

    public Set<Entity> getEntities() {
        return entities;
    }

    public Set<Relationship> getRelationships() {
        return relationships;
    }

    public Element[] getElementsArray() {
        Element[] entityArray = entities.toArray(new Element[0]);
        Element[] relationshipArray = relationships.toArray(new Element[0]);
        Element[] arr = new Element[entityArray.length + relationshipArray.length];
        for (int i = 0; i < entityArray.length; i++) {
            arr[i] = entityArray[i];
        }
        for (int i = 0; i < relationshipArray.length; i++) {
            arr[entityArray.length + i] = relationshipArray[i];
        }
        return arr;
    }

    public int size() {
        return entities.size() + relationships.size();
    }

    public boolean isEmpty() {
        return entities.isEmpty() && relationships.isEmpty();
    }
}
