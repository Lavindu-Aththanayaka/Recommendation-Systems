#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "data_loader.h"

// Compute Pearson Correlation between user u and user v
float pearson(RatingMatrix *R, int u, int v) {
    float num = 0.0f, den_u = 0.0f, den_v = 0.0f;

    for (int i = 0; i < R->num_items; i++) {
        // Only consider items BOTH users have rated
        if (R->ratings[u][i] != 0.0f && R->ratings[v][i] != 0.0f) {
            float du = R->ratings[u][i] - R->user_means[u];
            float dv = R->ratings[v][i] - R->user_means[v];
            num   += du * dv;
            den_u += du * du;
            den_v += dv * dv;
        }
    }

    if (den_u == 0.0f || den_v == 0.0f) return 0.0f;
    return num / (sqrtf(den_u) * sqrtf(den_v));
}

// Predict rating for user u on item i using top-K similar neighbors
float predict(RatingMatrix *R, float sim[][MAX_USERS], int u, int i, int K) {
    // Collect neighbors who have rated item i
    float weights[MAX_USERS];
    int   neighbors[MAX_USERS];
    int   count = 0;

    for (int v = 0; v < R->num_users; v++) {
        if (v != u && R->ratings[v][i] != 0.0f && sim[u][v] > 0.0f) {
            weights[count]   = sim[u][v];
            neighbors[count] = v;
            count++;
        }
    }

    if (count == 0) return R->user_means[u];

    // Simple top-K: sort by weight descending (insertion sort for clarity)
    for (int a = 1; a < count; a++) {
        float wt = weights[a]; int nb = neighbors[a];
        int b = a - 1;
        while (b >= 0 && weights[b] < wt) {
            weights[b+1] = weights[b]; neighbors[b+1] = neighbors[b]; b--;
        }
        weights[b+1] = wt; neighbors[b+1] = nb;
    }

    int use = (count < K) ? count : K;
    float num = 0.0f, den = 0.0f;
    for (int k = 0; k < use; k++) {
        int v = neighbors[k];
        num += weights[k] * (R->ratings[v][i] - R->user_means[v]);
        den += fabsf(weights[k]);
    }

    if (den == 0.0f) return R->user_means[u];
    return R->user_means[u] + num / den;
}

int main(int argc, char *argv[]) {
    const char *path = (argc > 1) ? argv[1] : "data/ratings.dat";

    RatingMatrix *R = load_data(path);

    // Allocate similarity matrix
    static float sim[MAX_USERS][MAX_USERS];

    double t_start = get_time();

    // Build full similarity matrix — THIS IS THE BOTTLENECK
    for (int u = 0; u < R->num_users; u++) {
        for (int v = u + 1; v < R->num_users; v++) {
            float s = pearson(R, u, v);
            sim[u][v] = s;
            sim[v][u] = s;  // symmetric
        }
    }

    double t_end = get_time();
    printf("Serial time: %.4f seconds\n", t_end - t_start);

    // Demo: predict rating for user 0 on item 9
    float pred = predict(R, sim, 0, 9, 10);
    printf("Predicted rating (user=0, item=9, K=10): %.2f\n", pred);

    // Save sim matrix for validation
    FILE *f = fopen("results/serial_sim.bin", "wb");
    fwrite(sim, sizeof(float), MAX_USERS * MAX_USERS, f);
    fclose(f);

    free_matrix(R);
    return 0;
}