from network_env import NetworkEnvironment
from rl_agent import NetworkRoutingEnv, train_rl_agent
from traffic_generator import TrafficGenerator
from baseline_routing import BaselineRouting

def main():
    # Create network
    env = NetworkEnvironment()
    traffic_gen = TrafficGenerator(env)
    traffic_gen.set_fixed_flows()
    flows = traffic_gen.get_flows()

    # Baseline
    baseline = BaselineRouting(env)
    baseline_results = baseline.simulate_traffic(flows)
    print("Baseline Results:", baseline_results)

    # RL Training
    source, target = (0,0), (3,3)  # Example
    rl_env = NetworkRoutingEnv(env, source, target)
    model = train_rl_agent(rl_env, total_timesteps=1000)

    # Test RL
    obs, _ = rl_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = rl_env.step(action)
    print("RL Path:", rl_env.path)

if __name__ == "__main__":
    main()