"""
Retrieve similar words and groups for 'unparliamentary' and variants of hypocrite.

Produces three sets of CSV files per query:

- The most similar identified clusters between the query, and the most similar 20
  features from that cluster.
- The top 100 most similar features across all features, regardless of cluster.
- The time distribution of that query (by parliament)

"""

import concurrent.futures as cf
import csv
import heapq
import multiprocessing as mp


from hyperreal.index import Index
from hansard_corpus import HansardCorpus
from pyroaring import BitMap

db = "tidy_hansard.db"
db_index = "tidy_hansard_index.db"

queries = [
    ["unparliamentary"],
    ["hypocrite", "hypocrisy", "hypocritical", "hypocrites"],
]

if __name__ == "__main__":

    corpus = HansardCorpus(db)

    mp_context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(mp_context=mp_context) as pool:
        idx = Index(db_index, corpus=corpus, pool=pool)

        for query in queries:
            expanded_query = [("text", w) for w in query]

            results = BitMap()

            for w in expanded_query:
                results |= idx[w]

            all_cluster_similarities = list(idx.pivot_clusters_by_query(results))

            with open(f"similarities_by_cluster_{query[0]}.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "word",
                        "similarity_to_query",
                        "cluster_id",
                        "cluster_similarity_to_query",
                    ]
                )

                for cluster_id, cluster_sim, features in all_cluster_similarities[:20]:
                    for (_, feature), similarity in features:
                        writer.writerow([feature, similarity, cluster_id, cluster_sim])

            k = 100
            top_k = [(0, ("", ""), -1, 0)] * k
            top_k_similarity = list(idx.pivot_clusters_by_query(results, top_k=k))

            for cluster_id, cluster_sim, features in top_k_similarity:
                for f, similarity in features:
                    heapq.heappushpop(top_k, (similarity, f, cluster_id, cluster_sim))

            with open(f"similarities_top_{k}_{query[0]}.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "word",
                        "similarity_to_query",
                        "cluster_id",
                        "cluster_similarity_to_query",
                    ]
                )

                for similarity, (_, feature), cluster_id, cluster_sim in sorted(
                    top_k, reverse=True
                ):
                    writer.writerow([feature, similarity, cluster_id, cluster_sim])

            # Timeline, by parliament
            values, totals, intersections = idx.intersect_queries_with_field(
                {"q": results}, "parl_no"
            )

            with open(f"timeline_{query[0]}.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(["parl_no", "total_speeches", "matching_speeches"])

                for v, t, i in zip(values, totals, intersections["q"]):
                    writer.writerow([v, t, i])
