"""Manual RoboSuite NutAssembly visualization smoke test.

This file is named like a pytest module for historical reasons, so all
interactive work stays inside ``main()`` to keep test collection side-effect
free.

Usage: python test_mujoco.py
"""
import numpy as np
import robosuite as suite


def main():
    """Render random actions in a registered single-arm environment."""

    env = suite.make("NutAssembly", robots="Panda", has_renderer=True)
    env.reset()

    print("RoboSuite smoke started — NutAssembly should be visible.")

    try:
        for _ in range(500):
            action = np.random.uniform(-1, 1, env.action_spec[0].shape[0])
            _, _, done, _ = env.step(action)
            env.render()
            if done:
                env.reset()
    finally:
        env.close()
    print("Smoke complete — environment closed successfully.")


if __name__ == "__main__":
    main()
