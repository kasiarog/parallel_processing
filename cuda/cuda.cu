#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <bitset>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

const int INF = numeric_limits<int>::max();
const int THREADS_PER_BLOCK = 256;

vector<vector<int>> generate_matrix(int n) {
    int rate = (int)(0.5 * (n * (n - 1) / 2));
    vector<vector<int>> matrix(n, vector<int>(n, INF));

    for (int i = 0; i < n; ++i) {
        matrix[i][i] = 0;
    }

    while (rate > 0) {
        int v1 = rand() % n;
        int v2 = rand() % n;
        if (v1 != v2 && matrix[v1][v2] == INF) {
            int value = 1 + rand() % 30;
            matrix[v1][v2] = value;
            matrix[v2][v1] = value;
            --rate;
        }
    }

    return matrix;
}

__device__ void dijkstra(int* matrix, int* dist, bool* visited, int n, int start) {
    for (int i = 0; i < n; i++) {
        dist[i] = INF;
        visited[i] = false;
    }
    dist[start] = 0;

    for (int count = 0; count < n - 1; count++) {
        int u = -1;
        for (int i = 0; i < n; i++) {
            if (!visited[i] && (u == -1 || dist[i] < dist[u])) {
                u = i;
            }
        }

        visited[u] = true;
        for (int v = 0; v < n; v++) {
            if (matrix[u * n + v] != INF && matrix[u * n + v] != 0 && dist[u] + matrix[u * n + v] < dist[v]) {
                dist[v] = dist[u] + matrix[u * n + v];
            }
        }
    }
}

__device__ void find_shortest_paths(int* matrix, int* shortest_paths, int n) {
    for (int start = 0; start < n; start++) {
        int* dist = new int[n];
        bool* visited = new bool[n];
        dijkstra(matrix, dist, visited, n, start);
        for (int i = 0; i < n; i++) {
            shortest_paths[start * n + i] = dist[i];
        }
        delete[] dist;
        delete[] visited;
    }
}

__device__ bool DFS_check_device(int* matrix, int n) {
    bool* visited = new bool[n];
    for (int i = 0; i < n; i++) {
        visited[i] = false;
    }

    auto DFS = [&](int v, auto& DFS_ref) -> void {
        visited[v] = true;
        for (int i = 0; i < n; ++i) {
            if (matrix[v * n + i] != INF && matrix[v * n + i] != 0 && !visited[i]) {
                DFS_ref(i, DFS_ref);
            }
        }
    };

    DFS(0, DFS);

    bool all_visited = true;
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            all_visited = false;
            break;
        }
    }

    delete[] visited;
    return all_visited;
}

__device__ bool check_the_limit_device(int* paths, int* new_paths, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (new_paths[i * n + j] > 1.5 * paths[i * n + j]) {
                return false;
            }
        }
    }
    return true;
}

__global__ void count_permutations_kernel(int* d_matrix, int* d_shortest_paths, int* d_result, int n, int edges_count, int* permutations, int* d_optimal_matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (1 << edges_count)) return;

    int* h_new_matrix = new int[n * n];
    for (int i = 0; i < n * n; i++) {
        h_new_matrix[i] = d_matrix[i];
    }

    int edge = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (h_new_matrix[i * n + j] != INF && h_new_matrix[i * n + j] != 0) {
                if (!(permutations[idx * edges_count + edge])) {
                    h_new_matrix[i * n + j] = INF;
                    h_new_matrix[j * n + i] = INF;
                }
                edge++;
            }
        }
    }

    if (DFS_check_device(h_new_matrix, n)) {
        int* d_new_paths = new int[n * n];
        find_shortest_paths(h_new_matrix, d_new_paths, n);

        if (check_the_limit_device(d_shortest_paths, d_new_paths, n)) {
            int edges_removed = 0;
            for (int j = 0; j < edges_count; j++) {
                if (!permutations[idx * edges_count + j]) {
                    edges_removed++;
                }
            }
            if (edges_removed > atomicMax(d_result, edges_removed)) {
                for (int i = 0; i < n * n; i++) {
                    d_optimal_matrix[i] = h_new_matrix[i];
                }
            }
        }

        delete[] d_new_paths;
    }

    delete[] h_new_matrix;
}

pair<int, vector<vector<int>>> delete_edges_brute_force(const vector<vector<int>>& matrix) {
    int n = matrix.size();
    int edges_count = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (matrix[i][j] != INF && matrix[i][j] != 0) {
                edges_count++;
            }
        }
    }
    edges_count /= 2;

    int* h_matrix;
    cudaMallocManaged(&h_matrix, n * n * sizeof(int));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_matrix[i * n + j] = matrix[i][j];
        }
    }

    int* d_shortest_paths;
    cudaMallocManaged(&d_shortest_paths, n * n * sizeof(int));

    int* h_shortest_paths = new int[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_shortest_paths[i * n + j] = INF;
        }
    }

    for (int start = 0; start < n; start++) {
        vector<int> dist(n, INF);
        vector<bool> visited(n, false);
        dist[start] = 0;

        for (int count = 0; count < n - 1; count++) {
            int u = -1;
            for (int i = 0; i < n; i++) {
                if (!visited[i] && (u == -1 || dist[i] < dist[u])) {
                    u = i;
                }
            }

            visited[u] = true;
            for (int v = 0; v < n; v++) {
                if (matrix[u][v] != INF && matrix[u][v] != 0 && dist[u] + matrix[u][v] < dist[v]) {
                    dist[v] = dist[u] + matrix[u][v];
                }
            }
        }

        for (int i = 0; i < n; i++) {
            h_shortest_paths[start * n + i] = dist[i];
        }
    }

    cudaMemcpy(d_shortest_paths, h_shortest_paths, n * n * sizeof(int), cudaMemcpyHostToDevice);

    int max_edges_removed = 0;

    int total_permutations = 1 << edges_count;
    int* permutations;
    cudaMallocManaged(&permutations, total_permutations * edges_count * sizeof(int));
    for (int i = 0; i < total_permutations; ++i) {
        for (int j = 0; j < edges_count; ++j) {
            permutations[i * edges_count + j] = (i >> j) & 1;
        }
    }

    int* d_result;
    cudaMallocManaged(&d_result, sizeof(int));
    cudaMemcpy(d_result, &max_edges_removed, sizeof(int), cudaMemcpyHostToDevice);

    int* d_optimal_matrix;
    cudaMallocManaged(&d_optimal_matrix, n * n * sizeof(int));

    clock_t start_time = clock();
    count_permutations_kernel<<<(total_permutations + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        h_matrix, d_shortest_paths, d_result, n, edges_count, permutations, d_optimal_matrix);
    cudaDeviceSynchronize();
    clock_t stop_time = clock();

    cudaMemcpy(&max_edges_removed, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Parallel execution time: " << (double)(stop_time - start_time) / CLOCKS_PER_SEC << " seconds." << endl;

    vector<vector<int>> optimal_matrix(n, vector<int>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            optimal_matrix[i][j] = d_optimal_matrix[i * n + j];
        }
    }

    cudaFree(permutations);
    cudaFree(h_matrix);
    cudaFree(d_shortest_paths);
    cudaFree(d_result);
    cudaFree(d_optimal_matrix);
    delete[] h_shortest_paths;

    return {max_edges_removed, optimal_matrix};
}

int main() {
    srand(time(nullptr));

    int n = 10;
    cout << "Number of vertices: " << n << "\n";
    vector<vector<int>> matrix = generate_matrix(n);

    cout << "Initial matrix:\n";
    for (const auto& row : matrix) {
        for (int val : row) {
            cout << (val == INF ? "INF" : to_string(val)) << " ";
        }
        cout << "\n";
    }

    clock_t start_time = clock();
    pair<int, vector<vector<int>>> result = delete_edges_brute_force(matrix);
    int bruteforce_solution = result.first;
    vector<vector<int>> optimal_matrix = result.second;
    clock_t stop_time = clock();

    cout << "\nOptimal number of edges removed: " << bruteforce_solution << "\n";
    if (bruteforce_solution > 0) {
        cout << "Optimal matrix:\n";
        for (const auto& row : optimal_matrix) {
            for (int val : row) {
                cout << (val == INF ? "INF" : to_string(val)) << " ";
            }
            cout << "\n";
        }
    } else {
        cout << "No edges could be removed while maintaining constraints.\n";
    }

    double elapsed_time = (double)(stop_time - start_time) / CLOCKS_PER_SEC;
    cout << "Brute force execution time: " << elapsed_time << " seconds\n";

    return 0;
}
