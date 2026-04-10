#ifndef QUIETPRIOR_H_INCLUDED
#define QUIETPRIOR_H_INCLUDED

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "position.h"
#include "types.h"

namespace Stockfish::QuietPrior {

enum class Tier : std::uint8_t {
    None = 0,
    Shallow = 1,
    Medium = 2,
    Full = 3,
};

struct MoveInfo {
    Move  move = Move::none();
    float logit = 0.0f;
    float prob = 0.0f;
    int   rank = 0;
};

struct PositionResult {
    bool                  ok = false;
    Key                   key = 0;
    float                 confidence = 0.0f;
    std::string           debugReason;
    int                   debugFenTokenCount = 0;
    int                   debugMappedMoves = 0;
    int                   debugForwardPasses = 0;
    std::vector<MoveInfo> moves;
};

struct Stats {
    std::uint64_t evaluations   = 0;
    std::uint64_t cacheHits     = 0;
    std::uint64_t cacheMisses   = 0;
    std::uint64_t legalMoves    = 0;
    std::uint64_t evalNanos     = 0;
    std::uint64_t gateChecks    = 0;
    std::uint64_t gatePasses    = 0;
    std::uint64_t gateMisses    = 0;
    std::uint64_t gateSignalSum = 0;
    std::uint64_t gateSignalMax = 0;
    std::uint64_t bonusMoves    = 0;
    std::uint64_t bonusAbsSum   = 0;
    std::int64_t  bonusSum      = 0;
    std::uint64_t bonusCalls    = 0;
    std::uint64_t topMoveChanges = 0;
};

bool enabled();
bool load(std::string_view path);
void unload();
PositionResult evaluate(const Position& pos, int plyFromRoot);
const MoveInfo* find_move(const PositionResult& result, Move move);
Stats stats();
void  reset_stats();
void  record_regime_gate(bool passed, int signal);
void  record_bonus_application(int moveCount, bool changedTopMove, int totalBonus, int totalAbsBonus);
int quiet_bonus_for_move(
  const PositionResult& result,
  Move                  move,
  int                   strength,
  float                 confidenceScale);

}  // namespace Stockfish::QuietPrior

#endif  // QUIETPRIOR_H_INCLUDED
