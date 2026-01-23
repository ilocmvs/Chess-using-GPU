#pragma once

#include "board.h"

__device__ __forceinline__ 
int add_move(Move* out, int count, int max,
                      int from, int to) {
  if (count >= max) return count; 
  out[count++] = Move{(int8_t)from, (int8_t)to, 0, 0};
  return count;
};

__device__ __forceinline__ 
bool in_bound(int index) {
    return index >= 0 && index < 64;
};

//pawn
__device__ int generate_pawn_moves(const Board* b, int i, int side,
                                     Move* out_moves, int count,
                                     int max_moves);
//king
__device__ int generate_king_moves(const Board* b, int i, int side,
                                     Move* out_moves, int count,
                                     int max_moves);
                                    