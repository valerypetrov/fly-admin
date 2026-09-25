"""Checks FlyBrain: speed, sugar -> feeding (Shiu et al.), and reward learning for grooming."""
import time

import numpy as np

from brain import FlyBrain


def trials(b, stim, n=6, ms=300, gap=200):
    # Mean and std of group rates over n trials, with a quiet gap so activity dies out.
    runs = []
    for _ in range(n):
        b.step({}, gap)
        runs.append(b.step(stim, ms))
    return {g: (np.mean([r[g] for r in runs]), np.std([r[g] for r in runs])) for g in runs[0]}


def show(label, res, keys=('feeding', 'grooming', 'escape', 'walking', 'dopamine_pam', 'kenyon', 'mbon')):
    print(f'  {label:28s} ' + '  '.join(f'{k}={res[k][0]:6.1f}+-{res[k][1]:4.1f}' for k in keys))


t0 = time.time()
b = FlyBrain()
print(f'load {time.time() - t0:.1f} s', b.info())

print('speed (wall seconds per simulated second, dt = 0.1 ms):')
for stim in ({}, {'sugar': 100}, {'looming': 200}, {'sugar': 200, 'bitter': 200, 'antennal_mechano': 200, 'looming': 200}):
    t0 = time.time()
    b.step(stim, 1000)
    print(f'  {str(stim):80s} {time.time() - t0:.2f} s, active neurons {int((b.counts > 0).sum())}')

print('sugar -> feeding (MN9), 6 trials x 300 ms each:')
for stim in ({}, {'sugar': 50}, {'sugar': 100}, {'sugar': 200}, {'bitter': 100}, {'bitter': 200},
             {'antennal_mechano': 150}, {'looming': 100}):
    show(str(stim), trials(b, stim))

print('learning: antennal_mechano 150 Hz, reward after grooming responses')
stim = {'antennal_mechano': 150}
before = trials(b, stim, n=10)
show('before', before)
for i in range(5):
    r = b.step(stim, 100)
    info = b.reward(1.0, chosen='groom')
    print(f'  reward {i + 1}: grooming={r["grooming"]:.1f} escape={r["escape"]:.1f} -> {info}')
after = trials(b, stim, n=10)
show('after', after)
print('  learned:', b.learned()['antennal_mechano'], 'bias:', b.action_bias())
print('controls after learning:')
show("{'sugar': 100}", trials(b, {'sugar': 100}))
show("{'looming': 100}", trials(b, {'looming': 100}))
snap = b.snapshot()
print('snapshot: raster entries', len(snap['raster']), 'slots', sum(e - s for s, e in snap['raster_slots'].values()),
      'dopamine', snap['dopamine'])
