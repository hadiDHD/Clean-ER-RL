package model;

import exp.ExpState;
import org.jgrapht.Graph;
import org.jgrapht.alg.clustering.KSpanningTreeClustering;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleGraph;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

public class HCS {

    public static List<ERModel> split(ERModel model) {
        {

            Graph<Element, DefaultEdge> g =
                    new SimpleGraph<>(DefaultEdge.class);

            for (Element e : model.elements) {
                g.addVertex(e);
            }

            for (Relationship r : model.relationships) {
                g.addEdge(r, r.a);
                g.addEdge(r, r.b);
            }

            for (Generalization gen : model.generalizations) {
                g.addEdge(gen.parent, gen.child);
            }

            System.out.println("Dividing");

            List<Graph<Element, DefaultEdge>> hcs = HCS(g);

            TreeSet<Graph<Element, DefaultEdge>> sortedSet = new TreeSet<>((o1, o2) -> o1.vertexSet().size() - o2.vertexSet().size());

            List<Graph<Element, DefaultEdge>> finalGraphSet = new ArrayList<>();

            for (Graph<Element, DefaultEdge> graph : hcs) {
                if (graph.vertexSet().size() < ExpState.MAX_ELEMENTS) {
                    sortedSet.add(graph);
                } else {
                    finalGraphSet.add(graph);
                }
            }

            System.out.println("Merging");

            while (sortedSet.size() > 2) {
                Graph<Element, DefaultEdge> smallest = sortedSet.pollFirst();
                Graph<Element, DefaultEdge> secondSmallest = sortedSet.pollFirst();
                if (smallest.vertexSet().size() + secondSmallest.vertexSet().size() <= ExpState.MAX_ELEMENTS) {
                    Graph<Element, DefaultEdge> merge = merge(smallest, secondSmallest);
                    sortedSet.add(merge);
                    System.out.println(smallest.vertexSet().size() + ", " + secondSmallest.vertexSet().size() + " -> " + merge.vertexSet().size());
                } else {
                    sortedSet.add(smallest);
                    sortedSet.add(secondSmallest);
                    break;
                }
            }

            finalGraphSet.addAll(sortedSet);

            List<ERModel> modelList = new ArrayList<>(hcs.size());

            for (Graph<Element, DefaultEdge> graph : finalGraphSet) {
                ERModel m = graphToER(graph, model);
                System.out.println(m);
                modelList.add(m);
            }

            return modelList;
        }
    }

    public static List<Graph<Element, DefaultEdge>> HCS(Graph<Element, DefaultEdge> g) {
        ArrayList<Graph<Element, DefaultEdge>> res = new ArrayList<>();
        if (g.vertexSet().size() < ExpState.MAX_ELEMENTS) {
            res.add(g);
        } else {
            StoerWagnerMinimumCut<Element, DefaultEdge> cut = new StoerWagnerMinimumCut<Element, DefaultEdge>(g);
            Graph<Element, DefaultEdge> H1 = new SimpleGraph<>(DefaultEdge.class);
            Graph<Element, DefaultEdge> H2 = new SimpleGraph<>(DefaultEdge.class);
            Set<Element> vertices = cut.minCut();
            for (Element v : vertices) {
                H1.addVertex(v);
            }
            for (Element v : g.vertexSet()) {
                if (!H1.containsVertex(v)) {
                    H2.addVertex(v);
                }
            }
            for (DefaultEdge e : g.edgeSet()) {
                if (H1.containsVertex(g.getEdgeSource(e)) && H1.containsVertex(g.getEdgeTarget(e))) {
                    H1.addEdge(g.getEdgeSource(e), g.getEdgeTarget(e));
                } else if (H2.containsVertex(g.getEdgeSource(e)) && H2.containsVertex(g.getEdgeTarget(e))) {
                    H2.addEdge(g.getEdgeSource(e), g.getEdgeTarget(e));
                }
            }
            System.out.println(g.vertexSet().size() + " -> " + H1.vertexSet().size() + ", " + H2.vertexSet().size());
            res.addAll(HCS(H1));
            res.addAll(HCS(H2));
        }
        return res;
    }

    public static ERModel graphToER(Graph<Element, DefaultEdge> g, ERModel er) {
        List<Entity> entities = new ArrayList<>();
        List<Relationship> relationships = new ArrayList<>();
        List<Generalization> generalizations = new ArrayList<>();
        for (Entity e : er.entities) {
            if (g.containsVertex(e)) {
                entities.add(e);
            }
        }
        for (Relationship r : er.relationships) {
            if (g.containsVertex(r)) {
                relationships.add(r);
            }
        }
        for (Generalization gen : er.generalizations) {
            if (g.containsVertex(gen.parent) && g.containsVertex(gen.child)) {
                generalizations.add(gen);
            }
        }
        ERModel res = new ERModel(entities, relationships, generalizations);
        res.init();
        return res;
    }

    public static Graph<Element, DefaultEdge> merge(Graph<Element, DefaultEdge> g1, Graph<Element, DefaultEdge> g2) {
        Graph<Element, DefaultEdge> union = new SimpleGraph<>(DefaultEdge.class);
        for (Element e : g1.vertexSet()) {
            union.addVertex(e);
        }
        for (Element e : g2.vertexSet()) {
            union.addVertex(e);
        }
        // no need to add edges, graphToER only needs vertices (elements).
        return union;
    }


    @Deprecated
    public static List<Graph<Element, DefaultEdge>> cluster(Graph<Element, DefaultEdge> g) {
        ArrayList<Graph<Element, DefaultEdge>> res = new ArrayList<>();
        if (g.vertexSet().size() <= 35) {
            res.add(g);
            return res;
        }
        KSpanningTreeClustering<Element, DefaultEdge> clustering = new KSpanningTreeClustering<>(g, 2);
        for (Set<Element> c : clustering.getClustering().getClusters()) {
            Graph<Element, DefaultEdge> H1 = new SimpleGraph<>(DefaultEdge.class);
            for (Element v : c) {
                H1.addVertex(v);
            }
            for (DefaultEdge e : g.edgeSet()) {
                if (H1.containsVertex(g.getEdgeSource(e)) && H1.containsVertex(g.getEdgeTarget(e))) {
                    H1.addEdge(g.getEdgeSource(e), g.getEdgeTarget(e));
                }
            }
            res.addAll(cluster(H1));
        }
        return res;
    }


}
