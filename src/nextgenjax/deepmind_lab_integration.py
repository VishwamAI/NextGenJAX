import os
import deepmind_lab
from dm_env import specs
import numpy as np
import traceback

def set_deepmind_lab_runfiles_path():
    lab_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    print(f"Setting DeepMind Lab runfiles path to: {lab_dir}")
    deepmind_lab.set_runfiles_path(os.path.join(lab_dir, 'lab'))
    print(f"Runfiles path set to: {deepmind_lab.runfiles_path()}")

set_deepmind_lab_runfiles_path()

class DeepMindLabIntegration:
    def __init__(self, level_name="seekavoid_arena_01", config=None):
        """
        Initialize the DeepMind Lab environment.

        Args:
            level_name (str): The name of the DeepMind Lab level to load.
            config (dict): Optional configuration dictionary for the environment.
        """
        self.level_name = level_name
        self.config = config if config is not None else {'width': '320', 'height': '240'}
        self.config['levelDirectory'] = os.path.abspath(os.path.join('lab', 'game_scripts', 'levels'))
        self.env = self._create_environment()

    def _create_environment(self):
        """
        Create and return a DeepMind Lab environment.

        Returns:
            deepmind_lab.Lab: The initialized DeepMind Lab environment.
        """
        observations = ['RGBD']  # Example observation
        print(f"Creating DeepMind Lab environment with:")
        print(f"  Level name: {self.level_name}")
        print(f"  Observations: {observations}")
        print(f"  Config: {self.config}")

        # Print the current working directory and the full path to the level file
        level_path = os.path.join(self.config['levelDirectory'], f'{self.level_name}.lua')
        print(f"Current working directory: {os.getcwd()}")
        print(f"Full path to level file: {level_path}")
        print(f"Level file exists: {os.path.exists(level_path)}")

        # Set the runfiles path before creating the environment
        deepmind_lab.set_runfiles_path(os.path.abspath('lab'))
        print(f"Full runfiles path: {os.path.abspath(deepmind_lab.runfiles_path())}")

        # Set the Lua package path
        lab_dir = os.path.abspath('lab')
        game_scripts_dir = os.path.join(lab_dir, 'game_scripts')
        factories_dir = os.path.join(game_scripts_dir, 'factories')
        lua_package_path = f"{game_scripts_dir}/?.lua;{factories_dir}/?.lua"
        os.environ['LUA_PATH'] = f"{lua_package_path};{os.environ.get('LUA_PATH', '')}"
        print(f"Set LUA_PATH: {os.environ['LUA_PATH']}")

        self.print_debug_info()

        try:
            env = deepmind_lab.Lab(
                os.path.basename(self.level_name),
                observations,
                config=self.config
            )
            print("DeepMind Lab environment created successfully")
            print(f"Lab environment attributes: {dir(env)}")
            return env
        except Exception as e:
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            raise

    def print_debug_info(self):
        print(f"Current working directory: {os.getcwd()}")
        print(f"Runfiles path: {deepmind_lab.runfiles_path()}")
        print(f"Level name: {self.level_name}")
        print(f"Config: {self.config}")
        level_path = os.path.join(self.config['levelDirectory'], f'{self.level_name}.lua')
        print(f"Full path to level file: {level_path}")
        print(f"Level file exists: {os.path.exists(level_path)}")

    def reset(self):
        """
        Reset the DeepMind Lab environment.

        Returns:
            dm_env.TimeStep: The initial time step of the environment.
        """
        return self.env.reset()

    def step(self, action):
        """
        Take a step in the DeepMind Lab environment.

        Args:
            action (np.ndarray): The action to take.

        Returns:
            dm_env.TimeStep: The time step resulting from the action.
        """
        return self.env.step(action)

    def observation_spec(self):
        """
        Get the observation specification of the environment.

        Returns:
            specs.Array: The observation specification.
        """
        return self.env.observation_spec()

    def action_spec(self):
        """
        Get the action specification of the environment.

        Returns:
            specs.Array: The action specification.
        """
        return self.env.action_spec()
