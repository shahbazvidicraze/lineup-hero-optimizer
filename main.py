# main.py (Lineup Optimizer Service)

import sys
import traceback # For detailed error logging
from flask import Flask, request, jsonify
# Modified pulp import to include necessary constants:
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
    LpStatusNotSolved,
    LpStatusInfeasible,
    LpStatusUnbounded,
    LpStatusUndefined
)

# --- Configuration ---
RESTRICTED_POSITION_PENALTY = 1000.0
NOT_PREFERRED_PENALTY = 10.0
BENCH_DEVIATION_WEIGHT = 1.0
SOLVER_TIME_LIMIT = 55 # Seconds, should be less than Laravel HTTP timeout

POSITIONS = ["CF", "LF", "RF", "SS", "1B", "2B", "3B", "P", "C", "OUT"]
MAIN_POSITIONS = ["CF", "LF", "RF", "SS", "1B", "2B", "3B", "P", "C"]
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
        if not isinstance(fixed_assignments, dict):
             raise ValueError("'fixed_assignments' field must be an object (dictionary).")

        actual_counts_input = input_data.get("actual_counts", {})
        if not isinstance(actual_counts_input, dict):
            raise ValueError("'actual_counts' field must be an object (dictionary).")

        game_innings = input_data.get("game_innings", 6)
        if not isinstance(game_innings, int) or game_innings <= 0:
            raise ValueError("'game_innings' must be a positive integer.")

        player_preferences = input_data.get("player_preferences", {})
        if not isinstance(player_preferences, dict):
             raise ValueError("'player_preferences' field must be an object (dictionary).")

        # Optional: Log less verbosely in production if needed
        print(f"Received Input:\n"
              f" Players: {players_str}\n"
              f" Game Innings: {game_innings}\n"
              f" Fixed Assign Type: {type(fixed_assignments)}\n"
              #f" Fixed Assign Keys: {list(fixed_assignments.keys())}\n" # Less verbose
              f" Actual Counts Type: {type(actual_counts_input)}\n"
              f" Preferences Type: {type(player_preferences)}\n"
              , file=sys.stderr)


        # --- Data Preprocessing ---
        actual_counts = {}
        for p_id in players_str:
            counts = actual_counts_input.get(p_id, {})
            if not isinstance(counts, dict):
                 print(f"Warning: Invalid actual_counts structure for player {p_id}. Expected dict, got {type(counts)}. Using empty.", file=sys.stderr)
                 counts = {}
            actual_counts[p_id] = {pos: float(counts.get(pos, 0)) for pos in POSITIONS}

        total_current = {
            p_id: sum(actual_counts[p_id].values()) for p_id in players_str
        }

        total_actual_out = sum(actual_counts[p_id].get("OUT", 0) for p_id in players_str)
        total_new_out = game_innings * max(0, (len(players_str) - len(MAIN_POSITIONS)))

        sum_T = sum(total_current.values()) + (game_innings * len(players_str))
        target_bench_ratio = (total_actual_out + total_new_out) / sum_T if sum_T > 0 else 0

        # --- PuLP Model Setup ---
        prob = LpProblem("LineupAssignment", LpMinimize)
        X = {
            p_id: {
                i: { pos: LpVariable(f"X_{p_id}_{i}_{pos}", cat=LpBinary) for pos in POSITIONS }
                for i in range(1, game_innings + 1)
            }
            for p_id in players_str
        }

        # --- Constraints ---
        # 1. One position per player per inning
        for p_id in players_str:
            for i in range(1, game_innings + 1):
                prob += lpSum(X[p_id][i][pos] for pos in POSITIONS) == 1, f"OnePos_{p_id}_{i}"

        # 2. One player per main position per inning
        for i in range(1, game_innings + 1):
            for pos in MAIN_POSITIONS:
                prob += lpSum(X[p_id][i][pos] for p_id in players_str) == 1, f"MainPos_{pos}_{i}"

        # 3. Fixed assignments
        for p_id, assignments in fixed_assignments.items(): # Safe due to input validation
            if p_id not in players_str: continue
            if not isinstance(assignments, dict):
                 print(f"Warning: Invalid fixed assignment value for player {p_id}. Expected dict, got {type(assignments)}. Skipping.", file=sys.stderr)
                 continue
            for inning_str, fixed_pos in assignments.items():
                try:
                    inning = int(inning_str)
                    # Ensure position is valid before adding constraint
                    if 1 <= inning <= game_innings and fixed_pos in POSITIONS:
                        prob += X[p_id][inning][fixed_pos] == 1, f"Fixed_{p_id}_{inning}_{fixed_pos}"
                    elif fixed_pos not in POSITIONS:
                        print(f"Warning: Invalid position '{fixed_pos}' in fixed assignment for player {p_id}, inning {inning}. Skipping.", file=sys.stderr)

                except (ValueError, KeyError) as e:
                     print(f"Warning: Invalid fixed assignment format for player {p_id}, inning {inning_str}, pos {fixed_pos}: {e}", file=sys.stderr)


        # --- Objective Function Terms ---
        # 4. Bench Time Fairness
        bench_counts = { p_id: lpSum(X[p_id][i]["OUT"] for i in range(1, game_innings + 1)) for p_id in players_str }
        d = {p_id: LpVariable(f"d_{p_id}", lowBound=0, cat=LpContinuous) for p_id in players_str}
        for p_id in players_str:
            target_bench_innings = target_bench_ratio * (total_current[p_id] + game_innings)
            # Use .get() for safety if actual_counts might miss OUT key somehow
            final_bench_innings = actual_counts[p_id].get("OUT", 0) + bench_counts[p_id]
            prob += final_bench_innings - target_bench_innings <= d[p_id], f"BenchDev1_{p_id}"
            prob += target_bench_innings - final_bench_innings <= d[p_id], f"BenchDev2_{p_id}"
        bench_deviation_objective = BENCH_DEVIATION_WEIGHT * lpSum(d[p_id] for p_id in players_str)

        # 5. Preference Penalties
        preference_penalty_objective = []
        for p_id in players_str:
            prefs = player_preferences.get(p_id, {})
            if not isinstance(prefs, dict):
                 print(f"Warning: Invalid structure for preferences for player {p_id}. Expected dict, got {type(prefs)}. Using empty.", file=sys.stderr)
                 prefs = {}
            preferred = set(prefs.get("preferred", [])) if isinstance(prefs.get("preferred"), list) else set()
            restricted = set(prefs.get("restricted", [])) if isinstance(prefs.get("restricted"), list) else set()

            for i in range(1, game_innings + 1):
                for pos in POSITIONS:
                    # Penalize restricted positions
                    if pos in restricted:
                        preference_penalty_objective.append(RESTRICTED_POSITION_PENALTY * X[p_id][i][pos])
                    # Penalize non-preferred positions (only if not OUT and not preferred)
                    elif pos != "OUT" and pos not in preferred:
                         preference_penalty_objective.append(NOT_PREFERRED_PENALTY * X[p_id][i][pos])

        # --- Combine Objective Terms ---
        prob += bench_deviation_objective + lpSum(preference_penalty_objective), "TotalObjective"

        # --- Solve the Model ---
        print("Solving MILP problem...", file=sys.stderr)
        # msg=1 shows solver output, msg=0 hides it
        solver = PULP_CBC_CMD(msg=0, timeLimit=SOLVER_TIME_LIMIT)
        prob.solve(solver)
        solve_status = LpStatus[prob.status] # Use imported LpStatus dictionary
        print(f"Solver status: {solve_status}", file=sys.stderr)

        # --- Process Results ---
        # Use imported constants directly
        if prob.status == LpStatusInfeasible:
             error_msg = "Optimization failed: Infeasible. Check conflicting fixed assignments or constraints."
             print(error_msg, file=sys.stderr)
             raise ValueError(error_msg) # Raise ValueError for 400 response
        elif prob.status not in [LpStatusOptimal, LpStatusUndefined]: # Allow Undefined (timeout with solution)
             print(f"Warning: Optimal solution not guaranteed ({solve_status}). Using best found solution.", file=sys.stderr)
        elif prob.status == LpStatusOptimal:
              print("Optimal solution found.", file=sys.stderr)

        # Check if variable values are None (can happen on timeout before feasible solution found)
        if any(value(X[p_id][i][pos]) is None for p_id in players_str for i in range(1, game_innings + 1) for pos in POSITIONS):
             error_msg = f"Optimization failed: Solver timed out ({SOLVER_TIME_LIMIT}s) before finding a feasible solution."
             print(error_msg, file=sys.stderr)
             raise ValueError(error_msg) # Raise ValueError for 400 response


        output_json = []
        for p_id in players_str:
            player_assignments = {"player_id": p_id, "innings": {}}
            for i in range(1, game_innings + 1):
                assigned_pos = "ERR" # Default error marker
                for pos in POSITIONS:
                    var_value = value(X[p_id][i][pos])
                    if var_value is not None and round(var_value) == 1:
                        assigned_pos = pos
                        break # Found the assigned position for this inning
                player_assignments["innings"][str(i)] = assigned_pos # Store inning key as string
            output_json.append(player_assignments)

        print("Optimization successful. Returning lineup.", file=sys.stderr)
        return jsonify(output_json)

    except ValueError as ve:
         # Client-side/Data validation errors
         print(f"Value Error: {ve}", file=sys.stderr)
         return jsonify({"error": str(ve), "input_received": input_data}), 400
    except Exception as e:
        # Server-side/Unexpected errors
        err_type = type(e).__name__
        err_msg = str(e)
        tb_str = traceback.format_exc() # Get full traceback
        print(f"Error during optimization: {err_type} - {err_msg}\nTraceback:\n{tb_str}", file=sys.stderr)
        return jsonify({"error": f"Internal optimization error: {err_type} - {err_msg}", "input_sent_by_client": input_data}), 500

# Run with: gunicorn --workers 3 --bind 0.0.0.0:5000 main:app
# Or: waitress-serve --host 0.0.0.0 --port 5000 main:app

# if __name__ == '__main__':
#     import os
#     port = int(os.environ.get("PORT", 8080))
#     app.run(host='0.0.0.0', port=port,debug=True)