import inspect
import rlutils.envs.fishEnvs as fenvs
for name, obj in inspect.getmembers(fenvs):
    if inspect.isclass(obj):
        print(obj.__name__)