#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "data_loader.h"

float pearson(RatingMatrix *R, int u, int v) {
    float num = 0.0f, den_u = 0.0f, den_v = 0.0f;
    for (int i = 0; i < R->num_items; i++) {
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


float predict(RatingMatrix *R, float *sim, int u, int item, int K) {
    int   N = R->num_users;
    float neighbors_w[N];
    float neighbors_r[N];
    int   count = 0;

    for (int v = 0; v < N; v++) {
        if (v != u &&
            R->ratings[v][item] != 0.0f &&
            sim[u * N + v] > 0.0f) {
            neighbors_w[count] = sim[u * N + v];
            neighbors_r[count] = R->ratings[v][item];
            count++;
        }
    }

    if (count == 0) return R->user_means[u];

    for (int a = 1; a < count; a++) {
        float wt = neighbors_w[a];
        float rt = neighbors_r[a];
        int b = a - 1;
        while (b >= 0 && neighbors_w[b] < wt) {
            neighbors_w[b+1] = neighbors_w[b];
            neighbors_r[b+1] = neighbors_r[b];
            b--;
        }
        neighbors_w[b+1] = wt;
        neighbors_r[b+1] = rt;
    }

    int   use = (count < K) ? count : K;
    float num = 0.0f, den = 0.0f;
    for (int k = 0; k < use; k++) {
        num += neighbors_w[k] * (neighbors_r[k] - R->user_means[u]);
        den += fabsf(neighbors_w[k]);
    }

    if (den == 0.0f) return R->user_means[u];
    return R->user_means[u] + num / den;
}


int main(int argc, char *argv[]) {
    const char *path     = (argc > 1) ? argv[1] : "data/ratings.dat";
    int         nthreads = (argc > 2) ? atoi(argv[2]) : 4;

    RatingMatrix *R = load_data(path);
    int N = R->num_users;

  
    float *sim = calloc(N * N, sizeof(float));
    if (!sim) {
        fprintf(stderr, "Cannot allocate sim matrix\n");
        return 1;
    }

    printf("Running OpenMP with %d threads...\n", nthreads);
    omp_set_num_threads(nthreads);

    double t_start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic) shared(sim, R)
    for (int u = 0; u < N; u++) {
        for (int v = u + 1; v < N; v++) {
            float s = pearson(R, u, v);
            sim[u * N + v] = s;
            sim[v * N + u] = s;
        }
    }

    double t_end = omp_get_wtime();

    printf("OpenMP (%d threads) time : %.4f seconds\n",
           nthreads, t_end - t_start);


    double serial_time = 0.8673; 
    double speedup     = serial_time / (t_end - t_start);
    printf("Speedup vs serial        : %.2fx\n", speedup);
    printf("Efficiency               : %.1f%%\n",
           (speedup / nthreads) * 100);

   
    printf("\n--------------------------------------------\n");

    int target_user = 0;  
    int target_item = 9;   
    int K           = 10;  

    
    if (R->ratings[target_user][target_item] != 0.0f) {
        printf("User %d already rated item %d: %.0f/5\n",
               target_user,
               target_item,
               R->ratings[target_user][target_item]);
    } else {
        float pred = predict(R, sim, target_user, target_item, K);
        printf("Predicted rating\n");
        printf("  User  : %d\n",   target_user);
        printf("  Item  : %d\n",   target_item);
        printf("  K     : %d\n",   K);
        printf("  Score : %.2f / 5.00\n", pred);
        printf("  Verdict: %s\n",
               pred >= 4.5 ? "*** Strongly Recommend" :
               pred >= 4.0 ? "**  Recommend" :
               pred >= 3.0 ? "*   Maybe" :
                             "    Not Recommended");
    }


   
    FILE *f = fopen("results/openmp_sim.bin", "wb");
    if (f) {
        fwrite(sim, sizeof(float), N * N, f);
        fclose(f);
        printf("Sim matrix saved → results/openmp_sim.bin\n");
    }

    free(sim);
    free_matrix(R);
    return 0;
}