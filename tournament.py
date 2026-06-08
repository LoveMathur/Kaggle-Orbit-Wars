from kaggle_environments import make
import main as production_agent
import basic_agent as baseline_agent
from statistics import mean

def run_tournament(num_games=20):
    print(f"🏆 Starting Tournament: Production vs Baseline ({num_games} games) 🏆\n")
    
    prod_wins = 0
    baseline_wins = 0
    draws = 0
    
    # Track raw scores to compute average performance margin
    prod_scores = []
    baseline_scores = []

    env = make("orbit_wars", debug=False) # Turn off debug prints for speed

    for game_idx in range(num_games):
        # Swap positions every game to eliminate map/seat bias
        if game_idx % 2 == 0:
            agents = [production_agent.simple_attacker, baseline_agent.simple_attacker]
            prod_player_idx = 0
            base_player_idx = 1
        else:
            agents = [baseline_agent.simple_attacker, production_agent.simple_attacker]
            prod_player_idx = 1
            base_player_idx = 0
            
        # Run the simulation
        env.run(agents)

        # Extract final step results (robust across kaggle_environments versions)
        final = env.steps[-1]
        prod_score = getattr(final[prod_player_idx], 'reward', 0) or 0
        base_score = getattr(final[base_player_idx], 'reward', 0) or 0
        
        prod_scores.append(prod_score)
        baseline_scores.append(base_score)
        
        # Determine Winner
        if prod_score > base_score:
            prod_wins += 1
            result = "PRODUCTION WIN"
        elif base_score > prod_score:
            baseline_wins += 1
            result = "BASELINE WIN"
        else:
            draws += 1
            result = "DRAW"
            
        print(f"Game {game_idx + 1:02d}: {result} | Prod: {prod_score} vs Base: {base_score}")

    # Calculate Metrics
    win_rate = (prod_wins / num_games) * 100
    avg_prod = mean(prod_scores) if prod_scores else 0.0
    avg_base = mean(baseline_scores) if baseline_scores else 0.0

    print("\n📊 FINAL BENCHMARK PERFORMANCE REPORT 📊")
    print("-" * 40)
    print(f"Production Agent Win Rate: {win_rate:.1f}%")
    print(f"Total Record: {prod_wins} Wins | {baseline_wins} Losses | {draws} Draws")
    print(f"Average Score Margin: {avg_prod:.1f} (Prod) vs {avg_base:.1f} (Base)")
    print("-" * 40)
    
    if win_rate >= 80:
        print("🔥 Status: Dominant. Ready for Kaggle leaderboard submission!")
    elif win_rate >= 60:
        print("📈 Status: Strong improvement, but can be optimized further.")
    else:
        print("⚠️ Status: Weak efficiency. Lead prediction might be over/under-correcting.")

# Run the evaluation harness
run_tournament(num_games=20)