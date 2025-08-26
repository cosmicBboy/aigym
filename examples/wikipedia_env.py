"""Example usage of the Web Gym environment."""

from aigym.env import WikipediaGymEnv


def main():
    # This runs 100 resets on the wikipedia environment
    env = WikipediaGymEnv(n_hops=10)

    for i in range(100):
        print(f"⭐️ Reset {i}")
        env.reset()
        print("travel path:")
        for url in env.travel_path:
            print(url)


if __name__ == "__main__":
    main()
