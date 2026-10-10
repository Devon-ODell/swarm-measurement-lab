#include <stdio.h>
#include <stdlib.h>

void bubble_sort(int *a, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (a[j] > a[j + 1]) {
                int t = a[j];
                a[j] = a[j + 1];
                a[j + 1] = t;
            }
        }
    }
}

int main(void) {
    char line[4096];
    int a[1024];
    int n = 0;

    if (fgets(line, sizeof line, stdin)) {
        char *p = line;
        while (n < (int)(sizeof a / sizeof a[0]) && *p) {
            char *end;
            long v = strtol(p, &end, 10);
            if (end == p)
                break;
            a[n++] = (int)v;
            p = end;
        }
    }

    bubble_sort(a, n);

    for (int i = 0; i < n; i++) {
        if (i)
            putchar(' ');
        printf("%d", a[i]);
    }
    if (n)
        putchar('\n');

    return 0;
}
