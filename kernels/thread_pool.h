#ifndef THREAD_POOL_H
#define THREAD_POOL_H

typedef struct thread_pool thread_pool_t;
typedef void (*pool_task_fn)(void* arg, int worker_id);

thread_pool_t* pool_create(int num_workers);
void pool_destroy(thread_pool_t* p);
void pool_submit(thread_pool_t* p, pool_task_fn fn, void* arg);
void pool_wait(thread_pool_t* p);
int pool_num_workers(const thread_pool_t* p);

#endif
