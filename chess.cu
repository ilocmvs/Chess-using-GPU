// gpu_chess_skeleton.cu
// Skeleton for "two GPUs compete in a chess-like game" assignment.
// This is NOT a full chess engine. It is an extensible scaffold.
//
// Build example:
//   nvcc -std=c++17 -O2 gpu_chess_skeleton.cu -o gpu_chess
//
// Run:
//   ./gpu_chess
//
// Notes:
// - Host keeps the authoritative board for simplicity.
// - Each turn: board -> GPU, GPU proposes move, move -> host, host applies it.
// - Two host threads bind to two GPUs and alternate turns via condition_variable.

#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      std::cerr << "CUDA error: " << cudaGetErrorString(err)                    \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";               \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// ------------------------------
// 1) Minimal chess data model
// ------------------------------

// A super-minimal piece encoding. You can replace with bitboards later.
enum Piece : int8_t {
  EMPTY = 0,

  WP = 1, WN, WB, WR, WQ, WK,
  BP = -1, BN = -2, BB = -3, BR = -4, BQ = -5, BK = -6
};

// Board is 8x8 = 64 squares
struct Board {
  int8_t squares[64];   // piece codes
  int8_t side_to_move;  // +1 white, -1 black
  // Add castling rights, en passant, halfmove clock, etc. if you want.
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

// --------------------------------------------
// 2) Device-side RNG helper (simple placeholder)
// --------------------------------------------
// For an assignment skeleton, a very simple LCG per thread is fine.
// Replace with curand if you want, but keep it simple for Coursera.
__device__ uint32_t lcg_next(uint32_t &state) {
  state = 1664525u * state + 1013904223u;
  return state;
}



// --------------------------------------------
// 3) Device-side hooks you will implement
// --------------------------------------------

// TODO: generate legal moves for side_to_move.
// For now we just pretend there are some candidates.
// In a real version, you'd fill a moves array and return count.
__device__ int generate_moves_device(const Board* b, Move* out_moves, int max_moves) {
  // PLACEHOLDER: produce some dummy moves so pipeline works.
  // Replace with real move gen.
  int count = 0;
  int side = b->side_to_move; // +1 white, -1 black
  //function to determine whether in board
  bool in_board = [] __device__ (int pos) {
    return pos >= 0 && pos < 64;
  };
  //add move, capture, etc.
  int add_move = [] __device__ (Move* out, int count, int max,
                        int from, int to) {
    if (count < max) out[count++] = Move{(int8_t)from, (int8_t)to, 0};
    return count;
  }
  //checkmate
  {

  }

  //all possible moves
  for (int i = 0; i < 64 && count < max_moves; i++) {
    int8_t p = b->squares[i];
    if (p == EMPTY) continue;
    if ((p > 0) != (side > 0)) continue; // not our piece

    //pawn
    if (p == WP || p == BP) {
      int dir = (side > 0) ? +8 : -8;

      // forward move (only if empty)
      int fwd = i + dir;
      if (in_bound(fwd) && b->squares[fwd] == EMPTY) {
        count = add_move(out_moves, count, max_moves, i, fwd);
      }

      // captures
      int capL = i + ((side > 0) ? +7 : -9);
      int capR = i + ((side > 0) ? +9 : -7);

      // You must also ensure file boundaries (avoid wrapping across board)
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
    }

    //king
    if (p == WP || p == BP) {
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
      if (file > 0 && in_bound(i + 1)) {
        if (b->squares[i + 1] == EMPTY || b->squares[i + 1] * side < 0)
          count = add_move(out_moves, count, max_moves, i, i + 1);
      }
      //up right
      if (file > 0 && in_bound(i - 7)) {
        if (b->squares[i - 7] == EMPTY || b->squares[i - 7] * side < 0)
          count = add_move(out_moves, count, max_moves, i, i - 7);
      }
      //bottom right
      if (file > 0 && in_bound(i + 9)) {
        if (b->squares[i + 9] == EMPTY || b->squares[i + 9] * side < 0)
          count = add_move(out_moves, count, max_moves, i, i + 9);
      }
    }
  }
  
  return count;
}

// TODO: evaluation function (material, piece-square tables, etc.)
__device__ int eval_board_device(const Board* b) {
  // PLACEHOLDER: material sum
  int score = 0;
  for (int i = 0; i < 64; i++) score += (int)b->squares[i];
  // from white perspective; multiply by side if you want side-relative.
  return score;
}

// TODO: search strategy kernel
// This kernel should pick one move for the current board.
// Different GPUs can run different kernels or strategies.
__global__ void choose_move_kernel(const Board* d_board,
                                   Move* d_best_move,
                                   int* d_found,
                                   uint32_t seed) {
  // One-block toy: thread0 generates moves, then pick one randomly.
  // Replace with parallel search (many threads evaluate many moves).

  __shared__ Move moves[256];
  __shared__ int move_count;
  __shared__ int scores[256];

  if (threadIdx.x == 0) {
    move_count = generate_moves_device(d_board, moves, 256);
    *d_found = (move_count > 0);
  }
  __syncthreads();
  if (move_count <= 0) return;

  int i = threadIdx.x;
  if (i < n) {
    Board tmp = *d_board;                 // <--- key: local copy
    apply_move_device(&tmp, moves[i]);
    int s = eval_board_device(&tmp);

    // Make "higher is better for side to move"
    s *= (int)b->side_to_move;

    scores[i] = s;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int best_i = 0;
    for (int k = 1; k < n; k++) {
      if (scores[k] > scores[best_i]) best_i = k;
    }
    int best_scores[256];
    int best_counts = 0
    for (int k = 1; k < n; k++) {
      if (scores[k] == best_i) best_scores[best_counts++] = k;
    }
    uint32_t st = seed ^ (uint32_t)clock64();
    int idx = (int)(best_scores[lcg_next(st) % (uint32_t)best_counts]);
    *d_best_move = moves[idx];
    *d_found = 1;
  }
}

// ------------------------------------------------------
// 4) Host-side chess hooks (apply move, end condition)
// ------------------------------------------------------

// TODO: apply move to board (must be correct if you care about chess)
void apply_move_host(Board& b, const Move& m) {
  // PLACEHOLDER: move piece, no legality checks
  int8_t p = b.squares[(int)m.from];
  b.squares[(int)m.from] = EMPTY;
  b.squares[(int)m.to] = p;
  b.side_to_move = -b.side_to_move;
}

// TODO: detect checkmate/stalemate/illegal, etc.
GameResult check_game_over_host(const Board& b, int ply_count) {
  // PLACEHOLDER: stop after N plies
  if (ply_count >= 80) return DRAW;
  //find king
  bool found_K = false;
  for (int i = 0; i < 64; i++) {
    if (b.side_to_move > 0 && b->squares[i] == BK
    || b.side_to_move < 0 && b->squares[i] == WK) {
      found_K = true;
      break;
    }
  }
  if (!found_K) {
    if (b.side_to_move > 0) return WHITE_WINS;
    else return BLACK_WINS;
  }
  return ONGOING;
}

// Initialize a standard-ish position (you can replace with real chess init)
Board make_initial_board() {
  Board b{};
  std::memset(b.squares, 0, sizeof(b.squares));
  b.side_to_move = +1;

  // SUPER MINIMAL setup: pawns only (to keep your early debugging simpler)
  for (int file = 0; file < 8; file++) {
    b.squares[8 + file] = WP;       // white pawns on rank 2
    b.squares[48 + file] = BP;      // black pawns on rank 7
  }
  // Optional: add kings so "game over" can be meaningful later.
  b.squares[4]  = WK;
  b.squares[60] = BK;

  return b;
}

// Optional: pretty print board for debugging
void print_board(const Board& b) {
  auto piece_char = [](int8_t p) -> char {
    switch (p) {
      case WP: return 'P'; case WN: return 'N'; case WB: return 'B';
      case WR: return 'R'; case WQ: return 'Q'; case WK: return 'K';
      case BP: return 'p'; case BN: return 'n'; case BB: return 'b';
      case BR: return 'r'; case BQ: return 'q'; case BK: return 'k';
      default: return '.';
    }
  };
  std::cout << (b.side_to_move > 0 ? "White" : "Black") << " to move\n";
  for (int r = 7; r >= 0; r--) {
    for (int f = 0; f < 8; f++) {
      std::cout << piece_char(b.squares[r * 8 + f]) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

// ------------------------------------------------------
// 5) Per-GPU "player" thread context + loop
// ------------------------------------------------------

struct PlayerGPUContext {
  int device_id;

  // Device allocations (per GPU)
  Board* d_board = nullptr;
  Move*  d_best_move = nullptr;
  int*   d_found = nullptr;

  cudaStream_t stream = nullptr;

  void init() {
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMalloc(&d_board, sizeof(Board)));
    CUDA_CHECK(cudaMalloc(&d_best_move, sizeof(Move)));
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(int)));
  }

  void destroy() {
    CUDA_CHECK(cudaSetDevice(device_id));
    if (d_board) CUDA_CHECK(cudaFree(d_board));
    if (d_best_move) CUDA_CHECK(cudaFree(d_best_move));
    if (d_found) CUDA_CHECK(cudaFree(d_found));
    if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
  }

  // Given a host board, ask this GPU to pick a move.
  // Returns true if a move was found.
  bool choose_move(const Board& h_board, Move& out_move, uint32_t seed) {
    CUDA_CHECK(cudaSetDevice(device_id));

    CUDA_CHECK(cudaMemcpyAsync(d_board, &h_board, sizeof(Board),
                               cudaMemcpyHostToDevice, stream));

    // One block is enough for skeleton. Expand later.
    choose_move_kernel<<<1, 128, 0, stream>>>(d_board, d_best_move, d_found, seed);
    CUDA_CHECK(cudaGetLastError());

    int found = 0;
    CUDA_CHECK(cudaMemcpyAsync(&found, d_found, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    if (found) {
      CUDA_CHECK(cudaMemcpyAsync(&out_move, d_best_move, sizeof(Move),
                                 cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return found != 0;
  }
};

// Shared state between threads
struct SharedGameState {
  Board board;
  int ply_count = 0;
  GameResult result = ONGOING;

  std::mutex m;
  std::condition_variable cv;
  int turn_side = +1; // +1 white, -1 black

  std::atomic<bool> stop{false};
};

// Thread function for one GPU player
void gpu_player_thread(PlayerGPUContext ctx, SharedGameState* gs) {
  ctx.init();

  std::mt19937 rng((unsigned)std::hash<std::thread::id>{}(std::this_thread::get_id()));

  while (!gs->stop.load()) {
    Board local_board{};
    int my_side = 0;

    {
      std::unique_lock<std::mutex> lk(gs->m);
      gs->cv.wait(lk, [&]{
        return gs->stop.load() || (gs->result == ONGOING && gs->turn_side == gs->board.side_to_move);
      });

      if (gs->stop.load() || gs->result != ONGOING) break;

      // This thread plays the side whose turn it is, if mapped to it.
      // Map: GPU0 = white, GPU1 = black (simple).
      my_side = (ctx.device_id == 0) ? +1 : -1;
      if (gs->board.side_to_move != my_side) {
        // Not my turn (shouldn’t happen if mapping & wait predicate align)
        continue;
      }
      local_board = gs->board;
    }

    // Choose move on GPU
    Move m{};
    uint32_t seed = (uint32_t)rng();
    bool found = ctx.choose_move(local_board, m, seed);

    {
      std::lock_guard<std::mutex> lk(gs->m);
      if (gs->result != ONGOING) break;

      if (!found) {
        // No moves found => treat as loss for side to move (placeholder)
        gs->result = (my_side > 0) ? BLACK_WINS : WHITE_WINS;
        gs->stop.store(true);
      } else {
        apply_move_host(gs->board, m);
        gs->ply_count++;
        gs->result = check_game_over_host(gs->board, gs->ply_count);

        // Switch turn
        gs->turn_side = gs->board.side_to_move;

        if (gs->result != ONGOING) {
          gs->stop.store(true);
        }
      }
    }

    gs->cv.notify_all();
  }

  ctx.destroy();
}

// --------------------------------------------
// 6) Main: spawn two GPU players, run game
// --------------------------------------------

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    std::cerr << "Need at least 2 CUDA devices for this assignment. Found: "
              << device_count << "\n";
    return 1;
  }

  SharedGameState gs;
  gs.board = make_initial_board();
  gs.turn_side = gs.board.side_to_move;

  std::cout << "Starting game on GPU0 (White) vs GPU1 (Black)\n";
  print_board(gs.board);

  PlayerGPUContext p0{0}, p1{1};

  std::thread t0(gpu_player_thread, p0, &gs);
  std::thread t1(gpu_player_thread, p1, &gs);

  // Kick off
  gs.cv.notify_all();

  // Monitor loop (optional)
  while (!gs.stop.load()) {
    {
      std::lock_guard<std::mutex> lk(gs.m);
      print_board(gs.board);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  t0.join();
  t1.join();

  // Final report
  std::cout << "Game ended after " << gs.ply_count << " plies.\n";
  switch (gs.result) {
    case WHITE_WINS: std::cout << "Result: White wins\n"; break;
    case BLACK_WINS: std::cout << "Result: Black wins\n"; break;
    case DRAW:       std::cout << "Result: Draw\n"; break;
    default:         std::cout << "Result: Ongoing? (unexpected)\n"; break;
  }

  return 0;
}
