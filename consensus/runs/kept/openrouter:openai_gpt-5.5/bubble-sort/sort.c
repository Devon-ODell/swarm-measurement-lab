#include <stdio.h>
#include <stdlib.h>

void bubble_sort(int *a, int n)
{
    for (int i = 0; i < n - 1; i++) {
        int swapped = 0;
        for (int j = 0; j < n - 1 - i; j++) {
            if (a[j] > a[j + 1]) {
                int tmp = a[j];
                a[j] = a[j + 1];
                a[j + 1] = tmp;
                swapped = 1;
            }
        }
        if (!swapped) {
            break;
        }
    }
}

int main(void)
{
    int *a = NULL;
    int n = 0;
    int cap = 0;
    int x;

    while (scanf("%d", &x) == 1) {
        if (n == cap) {
            int new_cap = cap == 0 ? 16 : cap * 2;
            int *tmp = realloc(a, (size_t)new_cap * sizeof(*a));
            if (tmp == NULL) {
                free(a);
                return 1;
            }
            a = tmp;
            cap = new_cap;
        }
        a[n++] = x;
    }

    bubble_sort(a, n);

    for (int i = 0; i < n; i++) {
        if (i > 0) {
            putchar(' ');
        }
        printf("%d", a[i]);
    }
    putchar('\n');

    free(a);
    return 0;
}
