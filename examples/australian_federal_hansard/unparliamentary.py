"""
Retrieve similar words and groups for 'unparliamentary' and variants of hypocrite.

Produces three sets of CSV files per query:

- The most similar identified clusters between the query, and the most similar 20
  features from that cluster.
- The top 100 most similar features across all features, regardless of cluster.
- The time distribution of that query (by parliament)


Additionally for incorporating diachronic components of analysis:

- Look at the set of most similar queries to the clusters, and also annotate with the
  time series of the number of speeches per parliament containing the intersection.

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

# Queries can only be used for the text column, and are a simple boolean OR (matching
# any of the words if present).
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

            # Timeline of query overall, by parliament
            values, totals, q_intersections = idx.intersect_queries_with_field(
                {"q": results}, "parl_no"
            )

            query_timeline = q_intersections["q"]

            with open(f"timeline_{query[0]}.csv", "w") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow(["parl_no", "total_speeches", "matching_speeches"])

                for v, t, i in zip(values, totals, query_timeline):
                    writer.writerow([v, t, i])

            all_cluster_similarities = list(idx.pivot_clusters_by_query(results))

            with open(f"similarities_by_cluster_{query[0]}.csv", "w") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow(
                    [
                        "word",
                        "similarity_to_query",
                        "cluster_id",
                        "cluster_similarity_to_query",
                    ]
                )

                for cluster_id, cluster_sim, features in all_cluster_similarities[:50]:
                    for (_, feature), similarity in features:
                        writer.writerow([feature, similarity, cluster_id, cluster_sim])

            with open(f"similar_clusters_for_annotation_{query[0]}.csv", "w") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow(["cluster_id", "most_similar_features", *range(1, 48)])

                for cluster_id, cluster_sim, features in all_cluster_similarities[:50]:
                    top_features = [f[0] for f in features]
                    cluster_repr = ", ".join(f[1] for f in top_features)

                    # Evaluate the OR of the top 20 most similar words and the
                    # intersection with the overall query to give a sense of how
                    # prevalent the words occur in the same speeches.
                    inter = BitMap()
                    for f in top_features:
                        inter |= idx[f]

                    inter &= results

                    # Intersection with time by parliament
                    values, totals, inter_results = idx.intersect_queries_with_field(
                        {"q": inter}, "parl_no"
                    )

                    normalised_by_q = [
                        i / total
                        for i, total in zip(inter_results["q"], query_timeline)
                    ]

                    writer.writerow((cluster_id, cluster_repr, *normalised_by_q))

            # Create concordances for the most similar words, in the places where the
            # word is used in a speech with unparliamentary.
            with open(f"similar_feature_concordances_{query[0]}.csv", "w") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow(
                    ["feature", "date", "house", "title", "url", "concordance"]
                )

                converter = idx.field_values["text"].segment_to_str

                for cluster_id, cluster_sim, features in all_cluster_similarities[:50]:
                    top_features = [f[0] for f in features]

                    for f in top_features:

                        to_match = {f[0]: [f[1]]}

                        # Document matches both the driving query and the similar word
                        # Sample at most 10 documents for each pair.
                        match_both = idx.sample_bitmap(idx[f] & results, 10)

                        # Construct the snippets matching the feature
                        for _, _, doc in idx.docs(match_both):

                            doc_features = idx.corpus.doc_to_features(doc)

                            matches = idx.match_doc_features(doc_features, to_match)
                            positions = sorted(
                                p
                                for positions in matches["text"].values()
                                for p in positions
                            )

                            window_size = 12
                            concordances = [
                                converter(
                                    doc_features["text"],
                                    max(p - window_size, 0),
                                    p + window_size + 1,
                                    highlight=to_match,
                                )
                                for p in positions
                            ]

                            other_fields = [
                                f[1],
                                doc["date"],
                                doc["house"],
                                doc["title"],
                                doc["url"],
                            ]

                            for line in concordances:
                                writer.writerow([*other_fields, line])

            k = 100
            top_k = [(0, ("", ""), -1, 0)] * k
            top_k_similarity = list(idx.pivot_clusters_by_query(results, top_k=k))

            for cluster_id, cluster_sim, features in top_k_similarity:
                for f, similarity in features:
                    heapq.heappushpop(top_k, (similarity, f, cluster_id, cluster_sim))

            with open(f"similarities_top_{k}_{query[0]}.csv", "w") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
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
