# main.py (Lineup Optimizer Service)

import sys
import traceback
import time
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
    LpStatusInfeasible,
    LpStatusUndefined,
    getSolver # To select a different solver if available
)

# --- Configuration ---
RESTRICTED_POSITION_PENALTY = 1000.0
NOT_PREFERRED_PENALTY = 10.0
BENCH_DEVIATION_WEIGHT = 1.0
SOLVER_TIME_LIMIT = 55 # Seconds

ALL_POSITIONS = ["CF", "LF", "RF", "SS", "1B", "2B", "3B", "P", "C", "OUT"]
MAIN_POSITIONS = ["CF", "LF", "RF", "SS", "1B", "2B", "3B", "P", "C"]
# --- End Configuration ---

app = Flask(__name__)

# --- OPTIMIZATION 1: Try to use a faster solver if installed ---
# HiGHS is a high-performance open-source solver that PuLP can use.
# If you run `pip install highspy` in your Python environment, this code will use it.
# Otherwise, it falls back to the default CBC.
SOLVER = getSolver('HiGHS_CMD', msg=0, timeLimit=SOLVER_TIME_LIMIT)
if not SOLVER.available():
    print("HiGHS solver not found. Falling back to default CBC solver.", file=sys.stderr)
    SOLVER = PULP_CBC_CMD(msg=0, timeLimit=SOLVER_TIME_LIMIT)


@app.route('/')
def hello():
    return "Hello Boss!"

@app.route('/optimize', methods=['POST'])
def optimize_lineup():
    start_time = time.time()
    input_data = None
    try:
        input_data = request.get_json()
        if input_data is None: raise ValueError("No JSON input received.")

        # --- Input Parsing and Validation ---
        players_str = input_data.get("players", [])
        if not players_str: raise ValueError("'players' field must be a non-empty list.")
        game_innings = input_data.get("game_innings", 6)
        fixed_assignments = input_data.get("fixed_assignments", {})
        actual_counts_input = input_data.get("actual_counts", {})
        player_preferences = input_data.get("player_preferences", {})

        print(f"Received {len(players_str)} players for {game_innings} innings.", file=sys.stderr)

        # --- Data Preprocessing ---
        actual_counts = {
            p_id: {pos: float(actual_counts_input.get(p_id, {}).get(pos, 0)) for pos in ALL_POSITIONS}
            for p_id in players_str
        }
        total_current = { p_id: sum(actual_counts[p_id].values()) for p_id in players_str }
        total_actual_out = sum(counts.get("OUT", 0) for counts in actual_counts.values())
        total_new_out = game_innings * max(0, (len(players_str) - len(MAIN_POSITIONS)))
        sum_T = sum(total_current.values()) + (game_innings * len(players_str))
        target_bench_ratio = (total_actual_out + total_new_out) / sum_T if sum_T > 0 else 0

        # --- PuLP Model Setup ---
        prob = LpProblem("LineupAssignment", LpMinimize)
        innings_range = range(1, game_innings + 1)

        # --- OPTIMIZATION 2: Create variables more efficiently ---
        X = LpVariable.dicts("pos", (players_str, innings_range, ALL_POSITIONS), cat=LpBinary)

        # List to hold all objective function terms, to be summed once at the end.
        objective_terms = []

        # --- OPTIMIZATION 3: More efficient preference lookups ---
        # Convert preference lists to sets for much faster "in" checks inside loops.
        prefs_as_sets = {}
        for p_id in players_str:
            player_pref = player_preferences.get(p_id, {})
            prefs_as_sets[p_id] = {
                'preferred': set(player_pref.get('preferred', [])),
                'restricted': set(player_pref.get('restricted', []))
            }

        # --- Constraints & Objective Function ---
        # We combine constraint and objective creation in the same loops to reduce overhead.

        # 1. One position per player per inning
        for p_id in players_str:
            for i in innings_range:
                prob += lpSum(X[p_id][i][pos] for pos in ALL_POSITIONS) == 1, f"OnePos_{p_id}_{i}"

        # 2. One player per main position per inning
        for i in innings_range:
            for pos in MAIN_POSITIONS:
                prob += lpSum(X[p_id][i][pos] for p_id in players_str) == 1, f"MainPos_{pos}_{i}"

        # 3. Fixed assignments
        for p_id, assignments in fixed_assignments.items():
            if p_id not in players_str: continue
            for inning_str, fixed_pos in assignments.items():
                inning = int(inning_str)
                if 1 <= inning <= game_innings and fixed_pos in ALL_POSITIONS:
                    prob += X[p_id][inning][fixed_pos] == 1, f"Fixed_{p_id}_{inning}"

        # 4. Enforce bench count per inning
        num_on_bench = max(0, len(players_str) - len(MAIN_POSITIONS))
        for i in innings_range:
            prob += lpSum(X[p_id][i]["OUT"] for p_id in players_str) == num_on_bench, f"BenchCount_{i}"

        # 5. Bench Time Fairness
        d = LpVariable.dicts("dev", players_str, lowBound=0)
        for p_id in players_str:
            bench_count = lpSum(X[p_id][i]["OUT"] for i in innings_range)
            target_bench = target_bench_ratio * (total_current[p_id] + game_innings)
            final_bench = actual_counts[p_id].get("OUT", 0) + bench_count
            prob += final_bench - target_bench <= d[p_id], f"BenchDev1_{p_id}"
            prob += target_bench - final_bench <= d[p_id], f"BenchDev2_{p_id}"
        # Add the deviation cost to our objective terms list
        objective_terms.append(BENCH_DEVIATION_WEIGHT * lpSum(d))

        # --- OPTIMIZATION 4: Pre-calculate all penalty costs ---
        # This is the most significant optimization. Instead of appending to a list inside
        # the loops, we generate all penalty terms with a single list comprehension.
        penalty_terms = []
        for p_id in players_str:
            player_prefs = prefs_as_sets[p_id]
            restricted = player_prefs['restricted']
            preferred = player_prefs['preferred']
            for i in innings_range:
                for pos in ALL_POSITIONS:
                    if pos in restricted:
                        penalty_terms.append(RESTRICTED_POSITION_PENALTY * X[p_id][i][pos])
                    elif pos != "OUT" and pos not in preferred:
                        penalty_terms.append(NOT_PREFERRED_PENALTY * X[p_id][i][pos])

        objective_terms.extend(penalty_terms)

        # --- Combine Objective Terms ---
        # Summing all objective terms together in one go is much faster.
        prob += lpSum(objective_terms), "TotalObjective"
        
        setup_time = time.time() - start_time
        print(f"Model setup took: {setup_time:.4f} seconds.", file=sys.stderr)

        # --- Solve the Model ---
        prob.solve(SOLVER)
        solve_status = LpStatus[prob.status]
        print(f"Solver status: {solve_status}", file=sys.stderr)
        
        if prob.status == LpStatusInfeasible:
            raise ValueError("Infeasible model. This may be due to conflicting fixed assignments.")
        if prob.status not in [LpStatusOptimal, LpStatusUndefined]:
             print(f"Warning: Optimal solution not found ({solve_status}). Using best found solution.", file=sys.stderr)
        if any(X[p_id][i][pos].varValue is None for p_id in players_str for i in innings_range for pos in ALL_POSITIONS):
             raise ValueError(f"Solver timed out after {SOLVER_TIME_LIMIT}s without finding a feasible solution.")

        # --- Process Results ---
        output_json = []
        for p_id in players_str:
            player_assignments = {"player_id": p_id, "innings": {}}
            for i in innings_range:
                for pos in ALL_POSITIONS:
                    if X[p_id][i][pos].varValue > 0.5: # More robust check than == 1
                        player_assignments["innings"][str(i)] = pos
                        break
            output_json.append(player_assignments)

        total_time = time.time() - start_time
        print(f"Optimization successful. Total time: {total_time:.4f} seconds.", file=sys.stderr)
        return jsonify(output_json)

    except ValueError as ve:
         print(f"Value Error: {ve}", file=sys.stderr)
         return jsonify({"error": str(ve), "input_received": input_data}), 400
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Error during optimization: {type(e).__name__} - {e}\nTraceback:\n{tb_str}", file=sys.stderr)
        return jsonify({"error": f"Internal optimization error: {type(e).__name__} - {e}"}), 500