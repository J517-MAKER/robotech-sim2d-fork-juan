// -*-c++-*-
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "bhv_mark_opponent.h"
#include "strategy.h"

#include "basic_actions/body_go_to_point.h"
#include "basic_actions/arm_point_to_point.h"
#include "basic_actions/neck_turn_to_ball_or_scan.h"
#include "basic_actions/basic_actions.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/player/world_model.h>
#include <rcsc/common/server_param.h>
#include <rcsc/common/logger.h>

#include <algorithm>

using namespace rcsc;

bool
Bhv_MarkOpponent::execute( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();
    const ServerParam & SP = ServerParam::i();

    // Solo actúa si el balón está en nuestra mitad y no hay compañero con el balón
    if ( wm.ball().pos().x > 5.0 ) return false;
    if ( wm.kickableTeammate() )    return false;

    const int role = Strategy::i().roleNumber( wm.self().unum() );

    // Solo para defensores de campo (roles 2-5)
    if ( role < 2 || role > 5 ) return false;

    const Vector2D goal_center( -SP.pitchHalfLength(), 0.0 );

    // ── Buscar el rival más peligroso sin marcar ──────────────────────────
    const PlayerObject * mark_target = nullptr;
    double best_score = -1.0e9;

    for ( const PlayerObject * opp : wm.opponents() )
    {
        if ( ! opp )             continue;
        if ( opp->isGhost() )    continue;
        if ( opp->goalie() )     continue;
        if ( opp->pos().x > 5.0 ) continue;  // solo en nuestra mitad

        // ── Anti-doble-marcaje: ¿hay un compañero defensor más cercano? ──
        bool claimed = false;
        for ( const PlayerObject * tm : wm.teammates() )
        {
            if ( ! tm ) continue;
            if ( tm->unum() == wm.self().unum() ) continue;

            const int tm_role = Strategy::i().roleNumber( tm->unum() );
            if ( tm_role < 2 || tm_role > 5 ) continue;  // solo defensores

            const double tm_dist   = tm->pos().dist( opp->pos() );
            const double self_dist = wm.self().pos().dist( opp->pos() );

            // Si el compañero está claramente más cerca, el rival ya está cubierto
            if ( tm_dist < self_dist - 1.5 )
            {
                claimed = true;
                break;
            }
        }
        if ( claimed ) continue;

        // ── Puntuación de peligro: más peligroso cuanto más cerca a nuestra portería ──
        double score = -opp->pos().dist( goal_center )       // cercanía a portería
                       - opp->pos().absY() * 0.2;            // penalizar posición muy lateral

        if ( score > best_score )
        {
            best_score  = score;
            mark_target = opp;
        }
    }

    if ( ! mark_target ) return false;

    // ── Calcular posición de marcaje: entre rival y portería ─────────────
    Vector2D opp_to_goal = goal_center - mark_target->pos();
    if ( opp_to_goal.r() > 0.001 )
        opp_to_goal.setLength( 1.8 );  // 1.8m hacia la portería desde el rival

    Vector2D mark_pos = mark_target->pos() + opp_to_goal;

    // Limitar para no entrar demasiado en la portería
    mark_pos.x = std::max( mark_pos.x, -SP.pitchHalfLength() + 2.0 );

    // ── Señalar al rival con el brazo (PointTo) ───────────────────────────
    agent->setArmAction( new Arm_PointToPoint( mark_target->pos() ) );

    dlog.addText( Logger::TEAM,
                  __FILE__": MarkOpponent unum=%d pos=(%.1f %.1f) mark_pos=(%.1f %.1f)",
                  mark_target->unum(),
                  mark_target->pos().x, mark_target->pos().y,
                  mark_pos.x, mark_pos.y );

    agent->debugClient().addMessage( "Mark%d", mark_target->unum() );
    agent->debugClient().setTarget( mark_pos );

    const double dist_thr = 0.8;
    if ( wm.self().pos().dist( mark_pos ) < dist_thr )
        return false;  // ya está en posición, dejar que el flujo normal gire hacia el balón

    Body_GoToPoint( mark_pos, dist_thr, SP.maxDashPower() ).execute( agent );

    if ( wm.ball().distFromSelf() < 15.0 )
        agent->setNeckAction( new Neck_TurnToBall() );
    else
        agent->setNeckAction( new Neck_TurnToBallOrScan( 0 ) );

    return true;
}
