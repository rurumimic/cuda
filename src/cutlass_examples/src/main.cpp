#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <iostream>

int main() {
  cutlass::half_t x = 2.25_hf;

  std::cout << x << "\n";

  return 0;
}
