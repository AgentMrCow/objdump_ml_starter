#include <stdio.h>
#include <stdlib.h>

static void swap(int* a, int* b) { int t = *a; *a = *b; *b = t; }

void bubble_sort(int* arr, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j+1]) {
                swap(&arr[j], &arr[j+1]);
            }
        }
    }
}

int main() {
    int n = 8;
    int arr[8] = {5,2,9,1,5,6,7,3};
    bubble_sort(arr, n);
    for (int i=0;i<n;++i) printf("%d ", arr[i]);
    printf("\n");
    return 0;
}
