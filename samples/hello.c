#include <stdio.h>
#include <stdint.h>

static int add(int a, int b) { return a + b; }
static int mult(int a, int b) { return a * b; }

int main() {
    int x = 7;
    int y = 5;
    int z = add(x, y);
    printf("hello %d\n", mult(z, 3));
    return 0;
}
