// -*-c++-*-
#ifndef BHV_MARK_OPPONENT_H
#define BHV_MARK_OPPONENT_H

#include <rcsc/player/soccer_action.h>

/*!
  \class Bhv_MarkOpponent
  \brief Defensive marking behavior with PointTo coordination.
  Each defender finds the most dangerous unmarked opponent,
  positions between them and the goal, and points arm at them.
*/
class Bhv_MarkOpponent
    : public rcsc::SoccerBehavior {
public:
    bool execute( rcsc::PlayerAgent * agent );
};

#endif
