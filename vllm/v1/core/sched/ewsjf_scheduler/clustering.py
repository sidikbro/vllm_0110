from sklearn.cluster import KMeans
from collections import deque
import numpy as np


# Cluster class holds sorted data points and tracks min/max for range-based logic
class Cluster:
    def __init__(self, data_points):
        self.data_points = sorted(data_points)
        self.min_len = self.data_points[0] if data_points else 0
        self.max_len = self.data_points[-1] if data_points else 0


# Main builder class for hybrid queue formation
class HybridQueueBuilder:
    def __init__(self, kmeans_k=3, min_cluster_size=50, significance_ratio=5.5,
                 max_queues=50, min_relative_range_factor=0.2, verbose=False):
        self.kmeans_k = kmeans_k
        self.min_cluster_size = min_cluster_size
        self.significance_ratio = significance_ratio
        self.max_queues = max_queues
        self.min_relative_range_factor = min_relative_range_factor
        self.verbose = verbose

    def create_cluster_from_data(self, data):
        return Cluster(data_points=data)

    def refine_cluster(self, cluster):
        final_clusters = []
        processing_queue = deque([cluster])

        while processing_queue:
            current = processing_queue.popleft()
            sorted_data = current.data_points

            if len(sorted_data) < self.min_cluster_size:
                final_clusters.append(current)
                continue

            # Compute range and dynamic minimum range threshold
            range_size = current.max_len - current.min_len
            mean_length = np.mean(sorted_data)
            min_range = mean_length * self.min_relative_range_factor

            if range_size < min_range:
                final_clusters.append(current)
                continue

            # Compute gaps and check for significant split
            gaps = [sorted_data[i+1] - sorted_data[i] for i in range(len(sorted_data) - 1)]
            if not gaps:
                final_clusters.append(current)
                continue

            max_gap = max(gaps)
            avg_gap = sum(gaps) / len(gaps)

            if self.verbose:
                print(f"[Refine] Cluster size: {len(sorted_data)}, range: {range_size}, max_gap: {max_gap:.2f}, avg_gap: {avg_gap:.2f}")

            if max_gap > self.significance_ratio * avg_gap:
                split_index = gaps.index(max_gap) + 1
                sub1 = sorted_data[:split_index]
                sub2 = sorted_data[split_index:]
                processing_queue.append(self.create_cluster_from_data(sub1))
                processing_queue.append(self.create_cluster_from_data(sub2))
            else:
                final_clusters.append(current)

        return final_clusters

    def compute_merge_score(self, cluster1, cluster2):
        # Composite score: lower density and smaller range → better merge candidate
        merged_data = cluster1.data_points + cluster2.data_points
        range_size = max(cluster2.max_len, cluster1.max_len) - min(cluster1.min_len, cluster2.min_len)
        density = len(merged_data) / max(1, range_size)
        score = (range_size ** 0.5) / (density + 1e-5)
        return score

    def prune_queues(self, queues):
        if len(queues) <= self.max_queues:
            return queues

        current_queues = sorted(queues, key=lambda c: c.min_len)

        while len(current_queues) > self.max_queues:
            best_score = float('inf')
            merge_index = -1

            # Evaluate all adjacent pairs for merge score
            for i in range(len(current_queues) - 1):
                score = self.compute_merge_score(current_queues[i], current_queues[i+1])
                if score < best_score:
                    best_score = score
                    merge_index = i

            if merge_index == -1:
                break  # No valid merge found

            if self.verbose:
                print(f"[Prune] Merging queues {merge_index} and {merge_index+1} with score {best_score:.2f}")

            # Merge selected pair
            cluster1 = current_queues.pop(merge_index)
            cluster2 = current_queues.pop(merge_index)
            merged_data = sorted(cluster1.data_points + cluster2.data_points)
            merged_cluster = self.create_cluster_from_data(merged_data)
            current_queues.insert(merge_index, merged_cluster)

        return sorted(current_queues, key=lambda c: c.min_len)

    def build_queues(self, data_points):
        # Step 1: KMeans clustering
        data_array = np.array(data_points).reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.kmeans_k, n_init='auto').fit(data_array)
        labels = kmeans.labels_

        initial_clusters = [
            self.create_cluster_from_data(data_array[labels == k].flatten().tolist())
            for k in range(self.kmeans_k)
        ]

        # Step 2: Refinement
        refined_queues = []
        for cluster in initial_clusters:
            refined_queues.extend(self.refine_cluster(cluster))

        # Step 3: Pruning
        final_queues = self.prune_queues(refined_queues)

        # Step 4: Metrics
        new_clusters = []
        for cluster in final_queues:
            lengths = cluster.data_points
            min_range = cluster.min_len
            max_range = cluster.max_len
            range_size = max_range - min_range if max_range > min_range else 1
            density = len(lengths) / range_size

            new_clusters.append({
                "boundaries": (min_range, max_range),
                "lengths": lengths,
                "count": len(lengths),
                "mean": np.mean(lengths),
                "std": np.std(lengths),
                "density": density,
                "range_size": range_size
            })

        if self.verbose:
            print(f"[Final] Total queues: {len(new_clusters)}")

        return new_clusters

    def build_queues_multiprocessing(self, queue, data_points):
        # Step 1: KMeans clustering
        data_array = np.array(data_points).reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.kmeans_k, n_init='auto').fit(data_array)
        labels = kmeans.labels_

        initial_clusters = [
            self.create_cluster_from_data(data_array[labels == k].flatten().tolist())
            for k in range(self.kmeans_k)
        ]

        # Step 2: Refinement
        refined_queues = []
        for cluster in initial_clusters:
            refined_queues.extend(self.refine_cluster(cluster))

        # Step 3: Pruning
        final_queues = self.prune_queues(refined_queues)

        # Step 4: Metrics
        new_clusters = []
        for cluster in final_queues:
            lengths = cluster.data_points
            min_range = cluster.min_len
            max_range = cluster.max_len
            range_size = max_range - min_range if max_range > min_range else 1
            density = len(lengths) / range_size

            new_clusters.append({
                "boundaries": (min_range, max_range),
                "lengths": lengths,
                "count": len(lengths),
                "mean": np.mean(lengths),
                "std": np.std(lengths),
                "density": density,
                "range_size": range_size
            })

        if self.verbose:
            print(f"[Final] Total queues: {len(new_clusters)}")

        queue.put(new_clusters)


if __name__ == "__main__":
    dataset = get_data_shuffled(num_samples=10000, min_range=1, max_range=6000,
                                data_path='/home/slava/data_for_sim.csv')
    data = np.array(dataset['in_len_tokens'])
    builder = HybridQueueBuilder(kmeans_k=7, max_queues=30, verbose=True)
    queues = builder.build_queues(data)

    for i, q in enumerate(queues):
        print(f"Queue {i + 1}: {q['count']} jobs, mean={q['mean']:.2f}, range={q['boundaries']}")
