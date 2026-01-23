#include "evaluate.cuh"
__device__ int eval_board_simplest(const Board* b) {
  int score = 0;
  for (int i = 0; i < 64; i++) score += (int)b->squares[i];
  return score;
}