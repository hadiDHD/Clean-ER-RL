package model;

import java.util.*;

public class ERModel {

    public List<Entity> entities;
    public List<Relationship> relationships;
    public List<Generalization> generalizations;
    transient public List<Element> elements;
    transient public Map<Element, Byte> indices;
    transient public Map<Entity, Set<Relationship>> entityToRelations;

    public ERModel(List<Entity> entities, List<Relationship> relationships) {
        this(entities, relationships, Collections.emptyList());
    }

    public ERModel(List<Entity> entities, List<Relationship> relationships, List<Generalization> generalizations) {
        this.entities = entities;
        this.relationships = relationships;
        this.generalizations = generalizations;
        init();
    }

    public void init() {
        if (generalizations == null) {
            generalizations = Collections.emptyList();
        }
        initElements();
        initIndicesZeroBased();
        initEntityToRelations();
    }

    public void initElements() {
        elements = new ArrayList<>(entities.size() + relationships.size());
        elements.addAll(entities);
        elements.addAll(relationships);
    }

    public void initIndicesOneBased() {
        indices = new HashMap<>(elements.size());
        for (int i = 0; i < elements.size(); i++) {
            Element element = elements.get(i);
            indices.put(element, (byte) (i + 1));
        }
    }

    public void initIndicesZeroBased() {
        indices = new HashMap<>(elements.size());
        for (int i = 0; i < elements.size(); i++) {
            Element element = elements.get(i);
            indices.put(element, (byte) (i));
        }
    }

    public void initEntityToRelations() {
        entityToRelations = new HashMap<>(entities.size());
        for (Entity e : entities) {
            entityToRelations.put(e, new HashSet<>());
        }
        for (Relationship r : relationships) {
            if (entityToRelations.containsKey(r.a))
                entityToRelations.get(r.a).add(r);
            if (entityToRelations.containsKey(r.b))
                entityToRelations.get(r.b).add(r);
        }
    }

    public void removeDisjointEntities() {
        List<Entity> disjoint = new ArrayList<>();
        E:
        for (Entity e : entities) {
            for (Relationship r : relationships) {
                if (r.a.equals(e) || r.b.equals(e)) {
                    continue E;
                }
            }
            disjoint.add(e);
        }
        entities.removeAll(disjoint);
    }

    @Override
    public String toString() {
        return "ERModel{" +
                "entities=" + Arrays.toString(entities.toArray()) +
                ", relationships=" + Arrays.toString(relationships.toArray()) +
                ", generalizations=" + Arrays.toString(generalizations.toArray()) +
                '}';
    }
}
