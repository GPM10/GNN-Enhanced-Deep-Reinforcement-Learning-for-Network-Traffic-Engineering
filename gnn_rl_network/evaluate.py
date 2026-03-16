from network_env import NetworkEnvironment
from traffic_generator import TrafficGenerator
from baseline_routing import BaselineRouting
from rl_agent import NetworkRoutingEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

def evaluate_methods():
    env = NetworkEnvironment()
    traffic_gen = TrafficGenerator(env)
    traffic_gen.set_fixed_flows()
    flows = traffic_gen.get_flows()

    baseline = BaselineRouting(env)

    # Shortest Path
    sp_results = baseline.simulate_traffic(flows)
    print("Shortest Path:", sp_results)

    # Random
    random_results = baseline.simulate_random_traffic(flows)
    print("Random:", random_results)

    # RL (assuming trained model)
    # For now, placeholder
    rl_results = sp_results  # Placeholder
    print("RL:", rl_results)

    # Plot
    methods = ['Shortest Path', 'Random', 'RL']
    latencies = [sp_results['total_latency'], random_results['total_latency'], rl_results['total_latency']]
    max_utils = [sp_results['max_utilization'], random_results['max_utilization'], rl_results['max_utilization']]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].bar(methods, latencies)
    ax[0].set_title('Total Latency')
    ax[1].bar(methods, max_utils)
    ax[1].set_title('Max Utilization')
    plt.show()

if __name__ == "__main__":
    evaluate_methods()