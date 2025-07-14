# main.py (Lineup Optimizer Service)

import sys
import traceback
from flask import Flask, request, jsonify
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    LpBinary,
    LpContinuous,
    value,
    PULP_CBC_CMD,
    LpStatus,
    LpStatusOptimal,
    LpStatusInfeasible
)

# --- Configuration ---
RESTRICTED_POSITION_PENALTY = 1000.0
NOT_PREFERRED_PENALTY = 10.0
BENCH_DEVIATION_WEIGHT = 1.0
SOLVER_TIME_LIMIT = 55 # Seconds

POSITIONS = ["CF", "LF", "RF", "SS", "1B", "2B", "3B", "P", "C", "OUT"]
MAIN_POSITIONS = ["CF", "LF", "RF", "SS", "1B", "2B", "3B", "P", "C"]
NUM_MAIN_POSITIONS = len(MAIN_POSITIONS) # Should be 9
# --- End Configuration ---


app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello Boss!"

@app.route('/optimize', methods=['POST'])
def optimize_lineup():
    input_data = None
    try:
        input_data = request.get_json()
        if input_data is None:
            raise ValueError("No JSON input received.")

        # --- Input Parsing and Validation ---
        players_str = input_data.get("players", [])
        if not players_str or not isinstance(players_str, list):
             raise ValueError("'players' field must be a non-empty list.")

        fixed_assignments = input_data.get("fixed_assignments", {})
        actual_counts_input = input_data.get("actual_counts", {})
        game_innings = input_data.get("game_innings", 6)
        player_preferences = input_data.get("player_preferences", {})

        # --- NEW: Separate core and extra players ---
        core_players = players_str[:NUM_MAIN_POSITIONS]
        extra_players = players_str[NUM_MAIN_POSITIONS:]

        print(f"Received {len(players_str)} players. Core: {len(core_players)}, Extra: {len(extra_players)}", file=sys.stderr)

        # --- Data Preprocessing (Only for Core Players) ---
        actual_counts = {}
        for p_id in core_players:
            counts = actual_counts_input.get(p_id, {})
            actual_counts[p_id] = {pos: float(counts.get(pos, 0)) for pos in POSITIONS}

        total_current = { p_id: sum(actual_counts[p_id].values()) for p_id in core_players }
        total_actual_out = sum(actual_counts[p_id].get("OUT", 0) for p_id in core_players)
        
        # Bench ratio calculation is now based on core players
        num_on_bench_core = max(0, len(core_players) - NUM_MAIN_POSITIONS)
        total_new_out = game_innings * num_on_bench_core
        sum_T = sum(total_current.values()) + (game_innings * len(core_players))
        target_bench_ratio = (total_actual_out + total_new_out) / sum_T if sum_T > 0 else 0

        # --- PuLP Model Setup (Only for Core Players) ---
        prob = LpProblem("LineupAssignment", LpMinimize)
        X = {
            p_id: {
                i: { pos: LpVariable(f"X_{p_id}_{i}_{pos}", cat=LpBinary) for pos in POSITIONS }
                for i in range(1, game_innings + 1)
            }
            for p_id in core_players
        }

        # --- Constraints (Applied to Core Players) ---
        # 1. One position per player per inning
        for p_id in core_players:
            for i in range(1, game_innings + 1):
                prob += lpSum(X[p_id][i][pos] for pos in POSITIONS) == 1, f"OnePos_{p_id}_{i}"

        # 2. One player per main position per inning
        for i in range(1, game_innings + 1):
            for pos in MAIN_POSITIONS:
                prob += lpSum(X[p_id][i][pos] for p_id in core_players) == 1, f"MainPos_{pos}_{i}"
        
        # 3. Explicitly define the number of benched players per inning
        for i in range(1, game_innings + 1):
            prob += lpSum(X[p_id][i]["OUT"] for p_id in core_players) == num_on_bench_core, f"BenchCount_{i}"

        # 4. Fixed assignments
        for p_id, assignments in fixed_assignments.items():
            if p_id not in core_players: continue
            for inning_str, fixed_pos in assignments.items():
                inning = int(inning_str)
                if 1 <= inning <= game_innings and fixed_pos in POSITIONS:
                    prob += X[p_id][inning][fixed_pos] == 1, f"Fixed_{p_id}_{inning}_{fixed_pos}"

        # --- Objective Function (For Core Players) ---
        # 5. Bench Time Fairness
        bench_counts = { p_id: lpSum(X[p_id][i]["OUT"] for i in range(1, game_innings + 1)) for p_id in core_players }
        d = {p_id: LpVariable(f"d_{p_id}", lowBound=0) for p_id in core_players}
        for p_id in core_players:
            target_bench = target_bench_ratio * (total_current[p_id] + game_innings)
            final_bench = actual_counts[p_id].get("OUT", 0) + bench_counts[p_id]
            prob += final_bench - target_bench <= d[p_id], f"BenchDev1_{p_id}"
            prob += target_bench - final_bench <= d[p_id], f"BenchDev2_{p_id}"
        bench_deviation_objective = BENCH_DEVIATION_WEIGHT * lpSum(d[p_id] for p_id in core_players)

        # 6. Preference Penalties
        preference_penalty_objective = []
        for p_id in core_players:
            prefs = player_preferences.get(p_id, {})
            preferred = set(prefs.get("preferred", []))
            restricted = set(prefs.get("restricted", []))
            for i in range(1, game_innings + 1):
                for pos in POSITIONS:
                    if pos in restricted:
                        preference_penalty_objective.append(RESTRICTED_POSITION_PENALTY * X[p_id][i][pos])
                    elif pos != "OUT" and pos not in preferred:
                         preference_penalty_objective.append(NOT_PREFERRED_PENALTY * X[p_id][i][pos])

        prob += bench_deviation_objective + lpSum(preference_penalty_objective), "TotalObjective"

        # --- Solve the Model ---
        print("Solving MILP problem for core players...", file=sys.stderr)
        solver = PULP_CBC_CMD(msg=0, timeLimit=SOLVER_TIME_LIMIT)
        prob.solve(solver)
        
        if prob.status == LpStatusInfeasible:
             raise ValueError("Lineup optimization failed due to conflicting constraints.")
        
        # --- Process Results ---
        output_json = []

        # Process results for core players
        for p_id in core_players:
            player_assignments = {"player_id": p_id, "isOut": False, "innings": {}}
            for i in range(1, game_innings + 1):
                assigned_pos = "ERR"
                for pos in POSITIONS:
                    if value(X[p_id][i][pos]) is not None and round(value(X[p_id][i][pos])) == 1:
                        assigned_pos = pos
                        break
                player_assignments["innings"][str(i)] = assigned_pos
            output_json.append(player_assignments)

        # --- NEW: Manually assign extra players to "OUT" ---
        for p_id in extra_players:
            player_assignments = {"player_id": p_id, "isOut": True, "innings": {}}
            for i in range(1, game_innings + 1):
                player_assignments["innings"][str(i)] = "OUT"
            output_json.append(player_assignments)

        print("Optimization successful. Returning final lineup.", file=sys.stderr)
        return jsonify(output_json)

    except ValueError as ve:
         print(f"Value Error: {ve}", file=sys.stderr)
         return jsonify({"error": str(ve), "input_received": input_data}), 400
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Error during optimization: {type(e).__name__} - {e}\n{tb_str}", file=sys.stderr)
        return jsonify({"error": "Internal optimization error", "details": str(e)}), 500