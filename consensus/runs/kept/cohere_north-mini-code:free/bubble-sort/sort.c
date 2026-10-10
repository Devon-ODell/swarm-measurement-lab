#include <stdio.h>
#include <stdlib.h>

void bubble_sort(int *a, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j + 1]) {
                int temp = a[j];
                a[j] = a[j + 1];
                a[j + 1] = temp;
            }
        }
    }
}

int main(void) {
    int capacity = 10;
    int *nums = malloc(capacity * sizeof(int));
    if (!nums) return 1;

    int count = 0;
    while (scanf("%d", &nums[count]) == 1) {
        count++;
        if (count >= capacity) {
            capacity *= 2;
            int *tmp = realloc(nums, capacity * sizeof(int));
            if (!tmp) {
                free(nums);
                return 1;
            }
            nums = tmp;
        }
    }

    bubble_sort(nums, count);

    for (int i = 0; i < count; i++) {
        printf("%d", nums[i]);
        if (i < count - 1) printf(" ");
    }
    printf("\n");

    free(nums);
    return 0;
}
