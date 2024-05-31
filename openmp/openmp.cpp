#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <bitset>
#include <string>
#include <omp.h>

using namespace std;

const int INF = numeric_limits<int>::max();

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

vector<vector<int>> find_shortest_paths(const vector<vector<int>>& matrix) {
    int n = matrix.size();
    vector<vector<int>> shortest_paths(n, vector<int>(n, INF));

    #pragma omp parallel for
    for (int start = 0; start < n; ++start) {
        vector<int> dist(n, INF);
        dist[start] = 0;
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        pq.push({0, start});

        while (!pq.empty()) {
            int u = pq.top().second;
            int d = pq.top().first;
            pq.pop();

            if (d > dist[u]) continue;

            for (int v = 0; v < n; ++v) {
                if (matrix[u][v] != INF && matrix[u][v] != 0) {
                    int alt = dist[u] + matrix[u][v];
                    if (alt < dist[v]) {
                        dist[v] = alt;
                        pq.push({alt, v});
                    }
                }
            }
        }
        #pragma omp critical
        shortest_paths[start] = dist;
    }

    return shortest_paths;
}

bool DFS_check(const vector<vector<int>>& matrix) {
    int n = matrix.size();
    vector<bool> visited(n, false);

    auto DFS = [&](int v, auto& DFS_ref) -> void {
        visited[v] = true;
        for (int i = 0; i < n; ++i) {
            if (matrix[v][i] != INF && matrix[v][i] != 0 && !visited[i]) {
                DFS_ref(i, DFS_ref);
            }
        }
    };

    DFS(0, DFS);

    return all_of(visited.begin(), visited.end(), [](bool v) { return v; });
}

bool check_the_limit(const vector<vector<int>>& paths, const vector<vector<int>>& new_paths) {
    int n = paths.size();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (new_paths[i][j] > 1.5 * paths[i][j]) {
                return false;
            }
        }
    }

    return true;
}

vector<vector<int>> make_matrix_from_permutation(const vector<vector<int>>& matrix, const vector<int>& permutation) {
    int n = matrix.size();
    vector<vector<int>> new_matrix = matrix;

    int edge = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (matrix[i][j] != INF && matrix[i][j] != 0) {
                if (!permutation[edge]) {
                    new_matrix[i][j] = INF;
                    new_matrix[j][i] = INF;
                }
                ++edge;
            }
        }
    }

    return new_matrix;
}

pair<int, vector<vector<int>>> delete_edges_brute_force(const vector<vector<int>>& matrix) {
    int n = matrix.size();
    int edges_count = accumulate(matrix.begin(), matrix.end(), 0, [](int sum, const vector<int>& row) {
        return sum + count_if(row.begin(), row.end(), [](int x) { return x != INF && x != 0; });
    }) / 2;
    vector<vector<int>> paths = find_shortest_paths(matrix);

    int max_edges_removed = 0;
    vector<vector<int>> optimal_matrix;

    #pragma omp parallel for
    for (int i = 0; i < (1 << edges_count); ++i) {
        vector<int> permutation(edges_count);
        for (int j = 0; j < edges_count; ++j) {
            permutation[j] = (i >> j) & 1;
        }

        vector<vector<int>> new_matrix = make_matrix_from_permutation(matrix, permutation);
        if (DFS_check(new_matrix)) {
            vector<vector<int>> new_paths = find_shortest_paths(new_matrix);
            if (check_the_limit(paths, new_paths)) {
                int edges_removed = count(permutation.begin(), permutation.end(), 0);
                #pragma omp critical
                {
                    if (edges_removed > max_edges_removed) {
                        max_edges_removed = edges_removed;
                        optimal_matrix = new_matrix;
                    }
                }
            }
        }
    }

    return {max_edges_removed, optimal_matrix};
}

int main() {
    srand(time(nullptr));

    int n = 9;
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
    auto [bruteforce_solution, optimal_matrix] = delete_edges_brute_force(matrix);
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
