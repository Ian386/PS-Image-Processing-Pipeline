#include "thread_pool.h"
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct task_node {
    pool_task_fn fn;
    void* arg;
    struct task_node* next;
} task_node_t;

struct thread_pool {
    int num_workers;
    pthread_t* workers;
    int* worker_ids;

    pthread_mutex_t mu;
    pthread_cond_t cond_work;
    pthread_cond_t cond_done;

    task_node_t* head;
    task_node_t* tail;
    int pending;
    int shutdown;
};

static void* worker_loop(void* arg) {
    void** ctx = (void**)arg;
    thread_pool_t* p = (thread_pool_t*)ctx[0];
    int wid = *(int*)ctx[1];
    free(ctx);

    for (;;) {
        pthread_mutex_lock(&p->mu);
        while (!p->shutdown && p->head == NULL) {
            pthread_cond_wait(&p->cond_work, &p->mu);
        }
        if (p->shutdown && p->head == NULL) {
            pthread_mutex_unlock(&p->mu);
            return NULL;
        }
        task_node_t* t = p->head;
        p->head = t->next;
        if (p->head == NULL) p->tail = NULL;
        pthread_mutex_unlock(&p->mu);

        t->fn(t->arg, wid);
        free(t);

        pthread_mutex_lock(&p->mu);
        p->pending--;
        if (p->pending == 0) pthread_cond_broadcast(&p->cond_done);
        pthread_mutex_unlock(&p->mu);
    }
}

thread_pool_t* pool_create(int num_workers) {
    if (num_workers < 1) num_workers = 1;
    thread_pool_t* p = (thread_pool_t*)calloc(1, sizeof(*p));
    p->num_workers = num_workers;
    p->workers = (pthread_t*)calloc(num_workers, sizeof(pthread_t));
    p->worker_ids = (int*)calloc(num_workers, sizeof(int));
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cond_work, NULL);
    pthread_cond_init(&p->cond_done, NULL);

    for (int i = 0; i < num_workers; i++) {
        p->worker_ids[i] = i;
        void** ctx = (void**)malloc(2 * sizeof(void*));
        ctx[0] = p;
        ctx[1] = &p->worker_ids[i];
        if (pthread_create(&p->workers[i], NULL, worker_loop, ctx) != 0) {
            fprintf(stderr, "pool_create: pthread_create failed\n");
            exit(1);
        }
    }
    return p;
}

void pool_destroy(thread_pool_t* p) {
    pthread_mutex_lock(&p->mu);
    p->shutdown = 1;
    pthread_cond_broadcast(&p->cond_work);
    pthread_mutex_unlock(&p->mu);
    for (int i = 0; i < p->num_workers; i++) pthread_join(p->workers[i], NULL);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cond_work);
    pthread_cond_destroy(&p->cond_done);
    free(p->workers);
    free(p->worker_ids);
    free(p);
}

void pool_submit(thread_pool_t* p, pool_task_fn fn, void* arg) {
    task_node_t* t = (task_node_t*)malloc(sizeof(*t));
    t->fn = fn;
    t->arg = arg;
    t->next = NULL;
    pthread_mutex_lock(&p->mu);
    if (p->tail) p->tail->next = t; else p->head = t;
    p->tail = t;
    p->pending++;
    pthread_cond_signal(&p->cond_work);
    pthread_mutex_unlock(&p->mu);
}

void pool_wait(thread_pool_t* p) {
    pthread_mutex_lock(&p->mu);
    while (p->pending > 0) pthread_cond_wait(&p->cond_done, &p->mu);
    pthread_mutex_unlock(&p->mu);
}

int pool_num_workers(const thread_pool_t* p) {
    return p->num_workers;
}
