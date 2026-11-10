"""Build agents by name, train with several environments, save and load generically."""

import relepy

print("available:", relepy.list_algorithms())

agent = relepy.make("a2c", "CartPole-v1", n_envs=4, n_steps=8, seed=0, verbose=0)
agent.fit(60_000)
print("A2C:", agent.evaluate(n_episodes=10))

# .relepy files are ZIP archives of JSON + .npy arrays: loading never runs code from the file.
agent.save("a2c_cartpole.relepy")
loaded = relepy.load("a2c_cartpole.relepy", "CartPole-v1", verbose=0)  # algorithm read from file
print("loaded:", loaded, loaded.evaluate(n_episodes=5))
