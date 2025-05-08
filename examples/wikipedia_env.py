"""Example usage of the Web Gym environment."""

from aigym.env import WikipediaGymEnv


def main():
    env = WikipediaGymEnv(n_hops=10)
    env.reset()
    print(env.target_url)
    print(env.start_url)
    print(env.travel_path)


if __name__ == "__main__":
    main()
