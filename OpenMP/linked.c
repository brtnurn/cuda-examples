#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#ifndef N
#define N 5
#endif
#ifndef FS
#define FS 38
#endif

struct node
{
    int data;
    int fibdata;
    struct node *next;
};

int fib(int n)
{
    int x, y;
    if (n < 2)
    {
        return (n);
    }
    else
    {
        x = fib(n - 1);
        y = fib(n - 2);
        return (x + y);
    }
}

void processwork(struct node *p)
{
    int n;
    n = p->data;
    p->fibdata = fib(n);
}

struct node *init_list(struct node *p)
{
    int i;
    struct node *head = NULL;
    struct node *temp = NULL;

    head = (struct node *)malloc(sizeof(struct node));
    p = head;
    p->data = FS;
    p->fibdata = 0;
    for (i = 0; i < N; i++)
    {
        temp = (struct node *)malloc(sizeof(struct node));
        p->next = temp;
        p = temp;
        p->data = FS + i + 1;
        p->fibdata = i + 1;
    }
    p->next = NULL;
    return head;
}

int main(int argc, char *argv[])
{
    double s_start, s_end;
    double p_start, p_end;
    struct node *p = NULL;
    struct node *temp = NULL;
    struct node *head = NULL;

    printf("Process linked list\n");
    printf("  Each linked list node will be processed by function 'processwork()'\n");
    printf("  Each ll node will compute %d fibonacci numbers beginning with %d\n", N, FS);

    p = init_list(p);
    head = p;

    s_start = omp_get_wtime();
    {
        while (p != NULL)
        {
            processwork(p);
            p = p->next;
        }
    }
    s_end = omp_get_wtime();

    p = head;
    printf("Compute Time (serial): %f seconds\n", s_end - s_start);

    while (p != NULL)
    {
        printf("%d : %d\n", p->data, p->fibdata);
        p = p->next;
    }
    p = head;

    p_start = omp_get_wtime();

    /*int count = 0;
    while (p != NULL)
    {
        count++;
        p = p->next;
    }
    p = head;
    struct node *arr[count];
    for (int i = 0; i < count; i++)
    {
        arr[i] = p;
        p = p->next;
    }
#pragma omp parallel for
    for (int i = 0; i < count; i++)
    {
        processwork(arr[i]);
    }*/

#pragma omp parallel
    {
#pragma omp single
        {
            struct node *my_node = p;
            while (my_node)
            {
#pragma omp task firstprivate(my_node)
                processwork(my_node);
                my_node = my_node->next;
            }
        }
    }

    p_end = omp_get_wtime();

    printf("Compute Time (parallel): %f seconds\n", p_end - p_start);

    p = head;
    while (p != NULL)
    {
        printf("%d : %d\n", p->data, p->fibdata);
        temp = p->next;
        free(p);
        p = temp;
    }
    free(p);

    return 0;
}
