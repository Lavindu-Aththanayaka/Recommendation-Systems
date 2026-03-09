#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "data_loader.h"

// Returns wall-clock time in seconds
double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

RatingMatrix* load_data(const char *filepath) {
    RatingMatrix *R = calloc(1, sizeof(RatingMatrix));
    if (!R) { fprintf(stderr, "Memory error\n"); exit(1); }

    // We use simple re-mapping arrays
    int user_map[100000] = {0}; // raw ID -> dense index
    int item_map[100000] = {0};
    int user_count = 0, item_count = 0;

    FILE *f = fopen(filepath, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filepath); exit(1); }

    int uid, iid, rating;
    long timestamp;

    while (fscanf(f, "%d\t%d\t%d\t%ld", &uid, &iid, &rating, &timestamp) == 4) {
        // Map raw ID to dense index
        if (user_map[uid] == 0) user_map[uid] = ++user_count;
        if (item_map[iid] == 0) item_map[iid] = ++item_count;

        int u = user_map[uid] - 1;
        int i = item_map[iid] - 1;

        if (u < MAX_USERS && i < MAX_ITEMS)
            R->ratings[u][i] = (float)rating;
    }
    fclose(f);

    R->num_users = user_count;
    R->num_items = item_count;

    compute_means(R);

    printf("Loaded: %d users, %d items\n", R->num_users, R->num_items);
    return R;
}

// Compute mean rating per user (only over rated items)
void compute_means(RatingMatrix *R) {
    for (int u = 0; u < R->num_users; u++) {
        float sum = 0.0f;
        int   count = 0;
        for (int i = 0; i < R->num_items; i++) {
            if (R->ratings[u][i] != 0.0f) {
                sum += R->ratings[u][i];
                count++;
            }
        }
        R->user_means[u] = (count > 0) ? (sum / count) : 0.0f;
    }
}

void free_matrix(RatingMatrix *R) {
    free(R);
}