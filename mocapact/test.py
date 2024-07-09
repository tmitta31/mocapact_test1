from mocapact import observables
from mocapact.sb3 import utils
# expert_path = "data/experts/CMU_009_12-165-363/eval_rsi/model"
expert_path = "mnt\expert_data\CMU_006_06-0-108\eval_rsi\model"
expert = utils.load_policy(expert_path, observables.TIME_INDEX_OBSERVABLES)

from mocapact.envs import tracking
from dm_control.locomotion.tasks.reference_pose import types
dataset = types.ClipCollection(ids=['CMU_006_06'], start_steps=[0], end_steps=[50])
env = tracking.MocapTrackingGymEnv(dataset)
obs, done = env.reset(), False
while not done:
    action, _ = expert.predict(obs, deterministic=True)
    obs, rew, done, _ = env.step(action)
    print(rew)