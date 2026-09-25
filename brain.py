"""FlyBrain: a spiking model of the whole adult fruit fly brain (FlyWire FAFB v783).

Leaky integrate-and-fire as in Shiu et al. 2024 (Nature). All 139,255 proofread neurons and
all 15,091,983 connections (54.5M synapses, >= 1 synapse per pair) are simulated.

Model (Shiu et al. parameters):
  dv/dt = (x - (v - v_rest)) / tau_m,  dx/dt = -x / tau_syn
  v_rest = v_reset = -52 mV, v_th = -45 mV, tau_m = 20 ms, tau_syn = 5 ms,
  refractory 2.2 ms, synaptic delay 1.8 ms, a presynaptic spike adds w to x of each target.
  The linear part is integrated exactly per step; dt = 0.1 ms (Shiu's value).
  A stimulated sensory neuron fires as a Poisson process at the requested rate (Shiu's
  Poisson kick is far above threshold, so each Poisson event is a spike).

Signs (Dale's law, one transmitter per neuron): argmax of the neuron's synapse-weighted mean
of the six per-connection NT probabilities. GABA and glutamate are inhibitory (-1);
acetylcholine, dopamine, serotonin and octopamine excitatory (+1), as in Shiu et al.
Rows with NaN probabilities are skipped; a neuron with none left uses its annotated top_nt.
Weight: w = sign * syn_count * 0.275 mV.

Groups come from nodes.parquet annotations (Schlegel et al. 2024), see GROUP_RULES.

Learning (three-factor): plastic synapses are all synapses onto the four motor groups.
After each step() a synapse's eligibility grows by pre_active * post_activity, where
post_activity is 1 if the target spiked, else its mean depolarisation / threshold.
Eligibility decays with TAU_ELIG of brain time. reward(d) fires the PAM neurons and grows
eligible excitatory weights by LR * d * eligibility * w0, capped at W_CAP * w0.
"""
import fnmatch
import json
import os
import sys
import urllib.request

import numpy as np

HF = 'https://huggingface.co/datasets/SLOP011/flywire-fafb-connectome/resolve/main/'
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NT_COLS = ['gaba', 'ach', 'glut', 'oct', 'ser', 'da']
INHIBITORY = {'gaba', 'glut'}

# (column, pattern) pairs; a neuron joins the group if any pair matches (fnmatch on the value).
GROUP_RULES = {
    'sugar': [('cell_sub_class', 'sugar/water')],                   # labellar sugar GRNs (LB3)
    'bitter': [('cell_sub_class', 'bitter')],                       # labellar bitter GRNs (LB1a/b)
    'antennal_mechano': [('cell_type', 'JO-C*'), ('cell_type', 'JO-E*')],  # JO-CE (Hampel 2015)
    'looming': [('hemibrain_type', 'LPLC2'), ('hemibrain_type', 'LC4')],  # looming detectors
    'feeding': [('cell_type', 'CB0701')],                           # MN9 pair (holds Shiu's MN9 id)
    'grooming': [('cell_type', 'DNg62'), ('cell_type', 'DNge078')],  # aDN1, aDN2 (Shiu's ids)
    'escape': [('cell_type', 'DNp01'), ('hemibrain_type', 'DNp02'), ('hemibrain_type', 'DNp04'),
               ('hemibrain_type', 'DNp06'), ('hemibrain_type', 'DNp11')],  # giant fiber + LC4 DNs
    'walking': [('cell_type', 'DNa01'), ('hemibrain_type', 'DNa02')],  # steering/walking DNs
    'dopamine_pam': [('hemibrain_type', 'PAM*')],                   # PAM dopaminergic cluster
    'kenyon': [('cell_class', 'Kenyon_Cell')],
    'mbon': [('cell_class', 'MBON')],
}
SENSES = ['sugar', 'bitter', 'antennal_mechano', 'looming']
MOTORS = ['feeding', 'grooming', 'escape', 'walking']
ACTION_TO_GROUP = {'feed': 'feeding', 'groom': 'grooming', 'escape': 'escape', 'walk': 'walking'}

W_SYN = 0.275      # mV per synapse
W_CAP = 3.0        # learned weight cap, as a multiple of the original weight
LR = 0.5           # weight growth per unit dopamine per unit eligibility
TAU_ELIG = 1000.0  # ms of brain time
TAU_DA = 300.0     # ms of brain time
BURST_MS = 20.0    # length of the PAM burst in reward()


def prepare(data_dir=DATA_DIR):
    """Download the parquet files once and convert them to compact numpy CSR files."""
    import pyarrow.parquet as pq
    os.makedirs(data_dir, exist_ok=True)
    for name in ('nodes.parquet', 'connections.parquet'):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print('downloading', name)
            urllib.request.urlretrieve(HF + name, path + '.part')
            os.replace(path + '.part', path)

    nodes = pq.read_table(os.path.join(data_dir, 'nodes.parquet')).to_pydict()
    root = np.array(nodes['root_id'], dtype=np.int64)
    order = np.argsort(root)
    n = len(root)

    pres, posts, syns, nt = [], [], [], np.zeros((len(NT_COLS), n))
    pf = pq.ParquetFile(os.path.join(data_dir, 'connections.parquet'))
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g)
        pre = order[np.searchsorted(root, t['pre'].to_numpy(), sorter=order)].astype(np.int32)
        post = order[np.searchsorted(root, t['post'].to_numpy(), sorter=order)].astype(np.int32)
        assert np.array_equal(root[pre], t['pre'].to_numpy()) and np.array_equal(root[post], t['post'].to_numpy())
        syn = t['syn_count'].to_numpy().astype(np.int32)
        prob = np.stack([t[c].to_numpy() for c in NT_COLS])
        ok = ~np.isnan(prob).any(0)  # about 1,250 rows have no NT prediction
        for k in range(len(NT_COLS)):
            nt[k] += np.bincount(pre[ok], weights=(syn * prob[k])[ok], minlength=n)
        pres.append(pre); posts.append(post); syns.append(syn)
    pre, post, syn = np.concatenate(pres), np.concatenate(posts), np.concatenate(syns)

    top, known = nt.argmax(0), nt.sum(0) > 0
    sign = np.where(np.isin(top, [NT_COLS.index(c) for c in INHIBITORY]), -1, 1).astype(np.int8)
    annotated = np.array([-1 if t in ('gaba', 'glutamate') else 1 for t in nodes['top_nt']], np.int8)
    sign[~known] = annotated[~known]
    idx = np.lexsort((post, pre))
    np.save(os.path.join(data_dir, 'csr_indptr.npy'),
            np.concatenate([[0], np.cumsum(np.bincount(pre, minlength=n))]).astype(np.int64))
    np.save(os.path.join(data_dir, 'csr_indices.npy'), post[idx])
    np.save(os.path.join(data_dir, 'csr_syn.npy'), np.minimum(syn[idx], 32767).astype(np.int16))
    np.save(os.path.join(data_dir, 'sign.npy'), sign)
    np.save(os.path.join(data_dir, 'root_id.npy'), root)

    groups, matched = {}, {}
    for name, rules in GROUP_RULES.items():
        hit = [i for i in range(n) if any(nodes[c][i] and fnmatch.fnmatchcase(nodes[c][i], p)
                                          for c, p in rules)]
        groups[name] = hit
        vals = {}
        for i in hit:
            key = ' / '.join(f'{c}={nodes[c][i]}' for c, p in rules
                             if nodes[c][i] and fnmatch.fnmatchcase(nodes[c][i], p))
            vals[key] = vals.get(key, 0) + 1
        matched[name] = vals
    with open(os.path.join(data_dir, 'groups.json'), 'w') as f:
        json.dump({'groups': groups, 'matched': matched, 'rules': GROUP_RULES}, f)
    tops = {NT_COLS[k]: int(((top == k) & known).sum()) for k in range(len(NT_COLS))}
    print(f'neurons {n}, connections {len(pre)}, synapses {int(syn.sum())}')
    print(f'transmitter by argmax {tops}, from top_nt fallback {int((~known).sum())}, '
          f'inhibitory {int((sign < 0).sum())}')
    for name, vals in matched.items():
        print(f'  {name:17s} {len(groups[name]):5d}  {vals}')


class FlyBrain:
    def __init__(self, data_dir=DATA_DIR, dt=0.1, seed=0):
        if not os.path.exists(os.path.join(data_dir, 'groups.json')):
            prepare(data_dir)
        load = lambda f: np.load(os.path.join(data_dir, f))
        self.indptr = load('csr_indptr.npy')
        self.indices = load('csr_indices.npy')
        sign = load('sign.npy')
        syn = load('csr_syn.npy')
        self.n = len(sign)
        self.n_synapses = int(syn.sum(dtype=np.int64))
        pre_of = np.repeat(np.arange(self.n, dtype=np.int32), np.diff(self.indptr))
        self.w = (syn * sign[pre_of] * W_SYN).astype(np.float32)
        del syn, pre_of
        with open(os.path.join(data_dir, 'groups.json')) as f:
            meta = json.load(f)
        self.groups = {k: np.array(v, dtype=np.int32) for k, v in meta['groups'].items()}
        self.matched = meta['matched']

        # Exact one-step propagators for the linear membrane and synapse equations.
        self.dt = dt
        tau_m, tau_s = 20.0, 5.0
        self.a_m = np.float32(np.exp(-dt / tau_m))
        self.a_s = np.float32(np.exp(-dt / tau_s))
        self.b = np.float32(tau_s / (tau_s - tau_m) * (np.exp(-dt / tau_s) - np.exp(-dt / tau_m)))
        self.v_th = np.float32(7.0)  # threshold above rest: -45 - (-52) mV
        self.delay = max(1, round(1.8 / dt))
        self.refr = max(1, round(2.2 / dt))

        self.rng = np.random.default_rng(seed)
        self.u = np.zeros(self.n, np.float32)   # v - v_rest
        self.x = np.zeros(self.n, np.float32)   # synaptic drive
        self.tmp = np.zeros(self.n, np.float32)
        self.hist = [np.empty(0, np.int32)] * max(self.delay, self.refr)  # ring of recent spikes
        self.hpos = 0
        self.t_ms = 0.0
        self.counts = np.zeros(self.n, np.int32)
        self.motor_idx = np.concatenate([self.groups[g] for g in MOTORS])
        self.motor_dep = np.zeros(len(self.motor_idx), np.float32)
        self.rates = {g: 0.0 for g in self.groups}
        self.stim = {}
        self.dopamine = 0.0
        self._setup_plasticity()
        self._setup_raster()
        self._probe_senses()

    # ---- simulation ----
    def _deliver(self, spk):
        # Add the outgoing weights of all neurons in spk to x.
        starts = self.indptr[spk]
        lens = self.indptr[spk + 1] - starts
        tot = int(lens.sum())
        if tot:
            pos = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(tot)
            np.add.at(self.x, self.indices[pos], self.w[pos])

    def _run(self, stim_idx, stim_p, steps):
        # Advance the network; returns raster entries [slot, t_ms] for sampled neurons.
        u, x, tmp, hist, H = self.u, self.x, self.tmp, self.hist, len(self.hist)
        raster, poisson = [], None
        self.counts[:] = 0
        self.motor_dep[:] = 0
        for k in range(steps):
            if len(stim_idx) and k % 1000 == 0:
                poisson = self.rng.random((min(1000, steps - k), len(stim_idx)), dtype=np.float32) < stim_p
            arriving = hist[(self.hpos - self.delay) % H]
            if len(arriving):
                self._deliver(arriving)
            np.multiply(u, self.a_m, out=u)
            np.multiply(x, self.b, out=tmp)
            u += tmp
            x *= self.a_s
            if poisson is not None:
                u[stim_idx[poisson[k % 1000]]] = 100.0
            for j in range(1, self.refr):  # neurons in their refractory period stay at rest
                r = hist[(self.hpos - j) % H]
                if len(r):
                    u[r] = 0.0
            self.motor_dep += np.minimum(u[self.motor_idx], self.v_th)
            spk = np.flatnonzero(u > self.v_th).astype(np.int32)
            if len(spk):
                u[spk] = 0.0
                x[spk] = 0.0
                self.counts[spk] += 1
                sl = self.slot_of[spk]
                raster.extend([int(s), round(self.t_ms, 1)] for s in sl[sl >= 0])
            hist[self.hpos % H] = spk
            self.hpos += 1
            self.t_ms += self.dt
        return raster

    def _group_rates(self, dur_s, names):
        return {g: float(self.counts[self.groups[g]].sum() / (len(self.groups[g]) * dur_s)) for g in names}

    def step(self, stim, duration_ms):
        """Poisson-stimulate sensory groups at stim[name] Hz for duration_ms; return rates (Hz) of all groups."""
        self.stim = {k: float(v) for k, v in stim.items() if v and v > 0}
        idx = [self.groups[k] for k in self.stim]
        p = [np.full(len(self.groups[k]), r * self.dt / 1000.0, np.float32) for k, r in self.stim.items()]
        stim_idx = np.concatenate(idx) if idx else np.empty(0, np.int32)
        stim_p = np.concatenate(p) if p else np.empty(0, np.float32)
        steps = max(1, int(round(duration_ms / self.dt)))
        self.raster = self._run(stim_idx, stim_p, steps)
        self.rates = self._group_rates(steps * self.dt / 1000.0, self.groups)
        self._update_traces(steps)
        self.dopamine *= float(np.exp(-steps * self.dt / TAU_DA))
        return dict(self.rates)

    # ---- learning ----
    def _setup_plasticity(self):
        # Plastic synapses: every synapse onto a neuron of a motor group.
        is_target = np.zeros(self.n, bool)
        is_target[self.motor_idx] = True
        self.pl_pos = np.flatnonzero(is_target[self.indices])
        self.pl_pre = (np.searchsorted(self.indptr, self.pl_pos, side='right') - 1).astype(np.int32)
        slot = np.full(self.n, -1, np.int32)
        slot[self.motor_idx] = np.arange(len(self.motor_idx))
        self.pl_slot = slot[self.indices[self.pl_pos]]  # index into motor_idx
        grp = np.concatenate([np.full(len(self.groups[g]), i) for i, g in enumerate(MOTORS)])
        self.pl_grp = grp[self.pl_slot]
        self.pl_w0 = self.w[self.pl_pos].copy()
        self.pl_exc = self.pl_w0 > 0
        self.elig = np.zeros(len(self.pl_pos), np.float32)
        self.pl_tot = [float(self.pl_w0[(self.pl_grp == gi) & self.pl_exc].sum()) or 1.0 for gi in range(len(MOTORS))]
        self.footprint = {}

    def _probe_senses(self, rate=150.0, ms=300.0):
        # Footprint: which plastic synapses each sense activates on its own, from a short run at start.
        for s in SENSES:
            self.step({s: rate}, ms)
            self.footprint[s] = (self.counts[self.pl_pre] > 0).astype(np.float32)
            self.step({}, ms)
        self.elig[:] = 0

    def _update_traces(self, steps):
        spiked = self.counts[self.motor_idx] > 0
        post = np.where(spiked, 1.0, np.clip(self.motor_dep / (steps * self.v_th), 0, 1)).astype(np.float32)
        pre_on = self.counts[self.pl_pre] > 0
        self.elig *= np.float32(np.exp(-steps * self.dt / TAU_ELIG))
        self.elig += pre_on * post[self.pl_slot]
        np.minimum(self.elig, 1.0, out=self.elig)

    def reward(self, dopamine, chosen=None):
        """Dopamine burst: fire the PAM neurons and strengthen eligible synapses onto motor neurons.
        chosen: optional action ('groom') or motor group ('grooming') to credit only that group."""
        d = float(max(0.0, dopamine))
        self.dopamine += d
        mask = self.pl_exc & (self.elig > 0.01)
        if chosen:
            mask &= self.pl_grp == MOTORS.index(ACTION_TO_GROUP.get(chosen, chosen))
        k = np.flatnonzero(mask)
        w0 = self.pl_w0[k]
        new = np.minimum(self.w[self.pl_pos[k]] + LR * d * self.elig[k] * w0, W_CAP * w0)
        self.w[self.pl_pos[k]] = new
        pam = self.groups['dopamine_pam']
        rate = min(300.0, 100.0 * d)
        steps = int(round(BURST_MS / self.dt))
        self.raster = getattr(self, 'raster', []) + self._run(
            pam, np.full(len(pam), rate * self.dt / 1000.0, np.float32), steps)
        self.rates.update(self._group_rates(BURST_MS / 1000.0, ('dopamine_pam', 'kenyon', 'mbon')))
        return {'dopamine': round(self.dopamine, 4), 'synapses': int(len(k)),
                'mean_gain': round(float((new / w0).mean()), 4) if len(k) else 1.0}

    def learned(self):
        """{sense: {motor group: input added by learning from that sense's neurons, as a share of
        the group's original excitatory input}}; 0 = unchanged."""
        extra = (self.w[self.pl_pos] - self.pl_w0) * self.pl_exc
        return {s: {g: round(float((self.footprint[s] * extra)[self.pl_grp == gi].sum()) / self.pl_tot[gi], 4)
                    for gi, g in enumerate(MOTORS)} for s in SENSES}

    def action_bias(self):
        """Learned gain per motor group for the current stimulus, weighted by stimulus rates."""
        table, tot = self.learned(), sum(self.stim.get(s, 0.0) for s in SENSES)
        if tot <= 0:
            return {g: 0.0 for g in MOTORS}
        return {g: round(sum(self.stim.get(s, 0.0) / tot * table[s][g] for s in SENSES), 4) for g in MOTORS}

    # ---- reporting ----
    def _setup_raster(self, total=200):
        # Sample ~total neurons, sharing slots fairly (small groups are taken whole).
        self.slot_of = np.full(self.n, -1, np.int32)
        rng = np.random.default_rng(1)
        quota, left = {}, total
        for i, g in enumerate(sorted(self.groups, key=lambda g: len(self.groups[g]))):
            quota[g] = min(len(self.groups[g]), left // (len(self.groups) - i))
            left -= quota[g]
        self.raster_slots, slot = {}, 0
        for g, ix in self.groups.items():
            pick = np.sort(rng.choice(ix, quota[g], replace=False))
            self.slot_of[pick] = np.arange(slot, slot + len(pick))
            self.raster_slots[g] = [slot, slot + len(pick)]
            slot += len(pick)
        self.raster = []

    def snapshot(self):
        return {'groups': {g: round(r, 3) for g, r in self.rates.items()},
                'raster': self.raster,
                'raster_slots': self.raster_slots,
                'dopamine': round(self.dopamine, 4),
                'learned': {s: {a: row[g] for a, g in ACTION_TO_GROUP.items()} for s, row in self.learned().items()},
                't_ms': round(self.t_ms, 1)}

    def info(self):
        return {'neurons': self.n, 'synapses': self.n_synapses, 'connections': len(self.indices),
                'groups': {g: len(ix) for g, ix in self.groups.items()}}


if __name__ == '__main__':
    if '--prep' in sys.argv:
        prepare()
