#include "piece_moves.cuh"

//pawn
__device__ int generate_pawn_moves(const Board* b, int i, int side,
                                     Move* out_moves, int count,
                                     int max_moves) {
    int dir = (side > 0) ? +8 : -8;

    // forward move
    int fwd = i + dir;
    if (in_bound(fwd) && b->squares[fwd] == EMPTY) {
      count = add_move(out_moves, count, max_moves, i, fwd);
    }
    // double move from starting position
    if (((side > 0) && i >= 8 && i < 16) || ((side < 0) && i >= 48 && i < 56)) {
      int dbl_fwd = i + 2 * dir;
      if (in_bound(dbl_fwd) && b->squares[dbl_fwd] == EMPTY && b->squares[fwd] == EMPTY) {
        count = add_move(out_moves, count, max_moves, i, dbl_fwd);
      }
    }

    // captures
    int capL = i + ((side > 0) ? +7 : -9);
    int capR = i + ((side > 0) ? +9 : -7);

    // file boundary checks
    int file = i % 8;
    if (file > 0 && in_bound(capL)) {
      if (b->squares[capL] * side < 0) {
        count = add_move(out_moves, count, max_moves, i, capL);
      }
    }
    if (file < 7 && in_bound(capR)) {
      if (b->squares[capR] * side < 0) {
        count = add_move(out_moves, count, max_moves, i, capR);
      }
    }
    return count;
}

//king
__device__ int generate_king_moves(const Board* b, int i, int side,
                                     Move* out_moves, int count,
                                     int max_moves) {
    //up
    if (in_bound(i - 8) && (b->squares[i - 8] == EMPTY || b->squares[i - 8] * side < 0)) {
      count = add_move(out_moves, count, max_moves, i, i - 8);
    }
    //down
    if (in_bound(i + 8) && (b->squares[i + 8] == EMPTY || b->squares[i + 8] * side < 0)) {
      count = add_move(out_moves, count, max_moves, i, i + 8);
    }
    int file = i % 8;
    //left
    if (file > 0 && in_bound(i - 1)) {
      if (b->squares[i - 1] == EMPTY || b->squares[i - 1] * side < 0)
        count = add_move(out_moves, count, max_moves, i, i - 1);
    }
    //up left
    if (file > 0 && in_bound(i - 9)) {
      if (b->squares[i - 9] == EMPTY || b->squares[i - 9] * side < 0)
        count = add_move(out_moves, count, max_moves, i, i - 9);
    }
    //bottom left
    if (file > 0 && in_bound(i + 7)) {
      if (b->squares[i + 7] == EMPTY || b->squares[i + 7] * side < 0)
        count = add_move(out_moves, count, max_moves, i, i + 7);
    }
    //right
    if (file < 7 && in_bound(i + 1)) {
      if (b->squares[i + 1] == EMPTY || b->squares[i + 1] * side < 0)
        count = add_move(out_moves, count, max_moves, i, i + 1);
    }
    //up right
    if (file < 7 && in_bound(i - 7)) {
      if (b->squares[i - 7] == EMPTY || b->squares[i - 7] * side < 0)
        count = add_move(out_moves, count, max_moves, i, i - 7);
    }
    //bottom right
    if (file < 7 && in_bound(i + 9)) {
      if (b->squares[i + 9] == EMPTY || b->squares[i + 9] * side < 0)
        count = add_move(out_moves, count, max_moves, i, i + 9);
    }
    return count;
}