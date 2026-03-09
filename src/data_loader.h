#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#define MAX_USERS  1000
#define MAX_ITEMS  2000

typedef struct {
    int    num_users;
    int    num_items;
    float  ratings[MAX_USERS][MAX_ITEMS]; // R[u][i], 0 = not rated
    float  user_means[MAX_USERS];         // mean rating per user
} RatingMatrix;

RatingMatrix* load_data(const char *filepath);
void          compute_means(RatingMatrix *R);
void          free_matrix(RatingMatrix *R);
double        get_time(void);

#endif