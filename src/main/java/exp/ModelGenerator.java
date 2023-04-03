package exp;

import model.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class ModelGenerator {

    public List<Element> elements;

    public List<Entity> entities;

    public List<Relationship> relationships;

    public Map<Element, Byte> indices;

    static final int MIN_RELATION = 1;

    static final int MIN_ENTITIES = 2;

    public ERModel generateModels() {
        Random r = new Random(System.currentTimeMillis());
        //TODO
        int numberOfEntities = r.nextInt(ExpState.MAX_ELEMENTS - MIN_RELATION + 1 - MIN_ENTITIES) + MIN_ENTITIES;
        int numberOfRelations = r.nextInt(ExpState.MAX_ELEMENTS + 1 - numberOfEntities - MIN_RELATION) + MIN_RELATION;
//        int numberOfRelations = 13;
//        int numberOfEntities = 12;
//        System.out.println("numberOfEntities: " + numberOfEntities);
//        System.out.println("numberOfRelations: " + numberOfRelations);

        entities = new ArrayList<>(numberOfEntities);
//        elements = new ArrayList<>(numberOfEntities + numberOfRelations);
//        indices = new HashMap<>(numberOfEntities + numberOfRelations);
        relationships = new ArrayList<>(numberOfRelations);

        for (int i = 0; i < numberOfEntities; i++) {
            Entity entity = new Entity("E" + (i + 1));
            entities.add(entity);
//            elements.add(entity);
//            indices.put(entity, (byte) (i + 1));
        }

        for (int i = 0; i < numberOfRelations; i++) {
            int indexA = r.nextInt(numberOfEntities);
            int indexB = indexA;
            while (indexB == indexA) {
                indexB = r.nextInt(numberOfEntities);
            }
            Relationship relationship = new Relationship("R" + (i + 1), entities.get(indexA), entities.get(indexB));
//            elements.add(relationship);
            relationships.add(relationship);
//            indices.put(relationship, (byte) (numberOfEntities + i + 1));
        }
        ERModel er = new ERModel(entities, relationships);
        er.initIndicesZeroBased();
        return er;
    }

}
