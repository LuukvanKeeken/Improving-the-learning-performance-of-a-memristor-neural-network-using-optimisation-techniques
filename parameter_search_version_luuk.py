import argparse
from subprocess import run

from memristor_nengo.extras import *

settings = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

for setting in settings:
    print(f"decay factor: {setting}")
    print(f"amount of settings: {len(settings)}")

    result = run(
                ["python3", "averaging_mPES_version_luuk.py", "-a", str(50), "-i", "sine", "-N", str(10), "-D", str(3),
                "-g", str(1e4), "-l", "mPES", "--optim_alg", str(2), "--SA_schedule", str(0), "--SA_starting_noise", str(0.3),
                "--SA_ending_noise", str(0.2), "--SA_exp_base", str(0.9995), "--pulse_levels", str(800), "--Momentum_decay_factor", str(setting)],
                capture_output=True,
                universal_newlines=True )

    print(result.stdout) 

