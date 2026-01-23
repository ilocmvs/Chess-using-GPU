#pragma once

#include <cstdint>
// Piece codes
enum Piece : int8_t {
  EMPTY = 0,

  WP = 1, WN = 2, WB = 3, WR = 4, WQ = 5, WK = 6,
  BP = -1, BN = -2, BB = -3, BR = -4, BQ = -5, BK = -6
};

// Board is 8x8 = 64 squares
struct Board {
  int8_t squares[64];   // piece codes
  int8_t side_to_move;  // +1 white, -1 black
};

struct Move {
  int8_t from;      // 0..63
  int8_t to;        // 0..63
  int8_t promo;     // 0 none, or piece code
  int8_t flags;     // optional: capture, castle, en-passant, etc.
};

// Game status returned by host checks
enum GameResult {
  ONGOING = 0,
  WHITE_WINS,
  BLACK_WINS,
  DRAW
};