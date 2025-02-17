#include <stdio.h>

#include "kernel.h"

int main(int argc, char *argv[]) {
  printf("Hello CUDA from CPU!\n");

  launchHelloCUDA();

  printf("Program completed successfully.\n");
  return 0;
}
