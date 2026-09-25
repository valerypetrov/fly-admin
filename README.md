# Fly Admin

A just-for-fun project. The whole adult fruit fly brain (the FlyWire FAFB v783 connectome: 139,255 neurons and
54.5 million synapses) runs as a spiking simulation and looks after a throwaway local ClickHouse server.
Problems in the database reach the fly as tastes, touches and looming shadows. The fly acts through its own
motor neurons. When an action fixes a problem, the fly gets a burst of dopamine and learns from it.

## Run it

```sh
git clone https://github.com/valerypetrov/fly-admin && cd fly-admin
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once
.venv/bin/python brain.py --prep     # once: download the fly brain (about 480 MB) from Hugging Face
.venv/bin/python run.py              # runs until Ctrl+C
```

It needs a `clickhouse` binary: put it on your PATH or set `CLICKHOUSE_BINARY=/path/to/clickhouse`
(get one with `curl https://clickhouse.com/ | sh`).

Open http://127.0.0.1:8765/ while it runs, or from another machine on the network use the address printed at start (e.g. http://my-computer.local:8765/). The UI listens on 0.0.0.0 and only shows data; pass `--host 127.0.0.1` to keep it local. The ClickHouse sandbox always stays on 127.0.0.1.

- **Make trouble** buttons in the Cluster card start a real issue in the sandbox (a table with too many parts, a slow
  query, or a burst of errors), so you can watch the fly react. They only trigger these three fixed issues.
- `--minutes N` stops after N minutes.
- `--record [PATH]` writes every event sent to the UI as one JSON line. Without a PATH the file is
  `runs/<timestamp>.jsonl`.
- Ctrl+C, or `kill`, stops the ClickHouse server and the UI cleanly.
- `python3 ui/server.py --demo` shows the UI with a fake fly and a fake cluster. It needs no brain and no database.
- `.venv/bin/python validate_brain.py` checks the brain on its own (about 40 s).
- `python3 cluster.py` checks the sandbox on its own: it creates each kind of issue, fixes it, and prints metrics
  before and after (about 50 s).

## What you see

- **Brain**: a front view of the fly brain. Each named neuron group glows with its firing rate: the senses at the
  top and middle, the mushroom body in the centre, and the four motor outputs at the bottom. When a reward
  arrives, the mushroom body and the dopamine neurons flash gold.
- **Spikes**: a scrolling 30 s raster of 200 sampled neurons, one row each, coloured by group. Gold lines mark
  rewards and violet ticks mark actions.
- **Cluster**: open issues with their age and severity, plus small trend charts for parts, the longest query,
  errors per second, queries per second and memory.
- **Decisions**: what the fly sensed, how strongly each motor group pushed ("drive"), which action won, the SQL it
  ran, and the dopamine it earned.
- **Learning**: time to fix each issue over the run, a per-kind "first half vs second half" headline, and a
  heatmap of what the fly has learned from each sense to each action.

## How the fly senses and acts

| Cluster state | Sense (Poisson input) | Right action | What the action does |
|---|---|---|---|
| nothing open | `sugar` 150 Hz | `feed` | nothing (all good) |
| error burst | `bitter` 50-200 Hz | `feed` / stay calm | nothing (the burst heals by itself) |
| too many parts | `antennal_mechano` 50-200 Hz | `groom` | `SYSTEM START MERGES` + `OPTIMIZE TABLE ... FINAL` on the dirtiest table |
| slow query | `looming` 50-200 Hz | `escape` | `KILL QUERY` for query ids starting with `fly_chaos_` |
| (never right) | | `walk` | `SYSTEM DROP MARK CACHE` |

Each loop takes about 0.5 s of wall time:

1. Read the metrics from the system tables. Work out the open issues and the sense rates. Rates grow with
   severity.
2. Run the brain for 200 ms of brain time with those inputs.
3. Give dopamine for every issue that closed since the last loop (see below).
4. Work out a drive for each action: the firing rate of its motor group plus a small learned bias. The groups are
   feeding (MN9), grooming (aDN1/aDN2), escape (giant fiber and other escape DNs), and walking (DNa01/DNa02).
5. Act only when the strongest drive beats its resting level by 8 Hz and the runner-up by 3 Hz. After any action
   the fly waits 2 s. It waits 10 s before repeating the same action, unless a new issue opened in the meantime.

No rule links a sense to an action. Sugar drives MN9 and looming drives the giant fiber because the connectome
wires them that way, as in Shiu et al. 2024. Antennal touch drives the grooming neurons only weakly at first.

## How dopamine learning works

- **Eligibility**: every synapse onto a motor neuron keeps an eligibility trace. It grows whenever the presynaptic
  neuron fired and the motor neuron fired, or was pushed toward threshold, in the same window. It decays with a
  1 s time constant of brain time.
- **Reward**: when an issue closes, `run.py` credits the last action the fly took while that issue was open. It
  calls `brain.reward(dopamine, chosen=action)`. The brain fires the PAM dopamine neurons for 20 ms, then
  strengthens the eligible excitatory synapses onto that action's motor neurons in proportion to the dopamine.
  Weights are capped at 3 times their original value.
- **Dopamine amount**: `exp(-time_to_fix / 30 s)`, never below 0.2. A fix in 1 s earns about 0.97 and a fix in
  15 s about 0.6. An error burst that heals by itself earns 0.3 for `feed`, but only if the fly did nothing
  pointless while it lasted. A pointless action is a groom or escape with no matching issue open, or a walk.
- **Heatmap**: for each sense and action, it shows the input that learning added, from the neurons that sense
  activates, as a share of the motor group's original excitatory input. At the start the brain probes each sense
  alone for 300 ms to find those neurons. The same number, weighted by the current input, is the "learned bias"
  in the drive (10 Hz per unit).

## Measured: a 4-minute run

From `runs/20260924-162950.jsonl`, with the code as it is now:

- **Issues**: 26 were created (16 slow queries, 6 with too many parts, 4 error bursts). 25 closed with a reward.
  One error burst closed without one, because the fly made a pointless `escape` while it was open.
- **Actions**: 39 in total (18 escape, 15 feed, 6 groom, 0 walk) and none failed. Total dopamine delivered: 21.4.
- **Too many parts** fixed in 15.1, 9.7, 4.1, 0.5, 3.0 and 2.0 s. The median went from 9.7 s in the first half of
  the run to 2.0 s in the second.
  - This is learning in the network. The fly first groomed at 39 parts, with antennal input at 164 Hz. In its
    last four fixes it groomed at 10-17 parts, with input at 80-101 Hz.
  - In the last three grooms, input at 80-95 Hz made the grooming neurons fire 12.5-15 Hz. The untrained brain
    fires 0 Hz at 80 Hz and about 2 Hz at 125 Hz.
  - The learned bias added at most 1.5 Hz to any groom drive.
- **Slow queries** were fixed in 0.6-2.0 s the whole time (median 0.8 s). Escape is innate, so there was nothing
  to learn. Rewards still pushed the escape rate from about 85 Hz to about 235 Hz, where it levelled off.
- **Error bursts** healed by themselves after 26-40 s. The fly stayed calm, because bitter input almost
  completely silences all four motor groups.

The same parts trend showed up in two more runs: 14.6, 9.7, 1.0, 0.5, 1.0 s in an earlier 4-minute run, and 13.8,
8.2, 2.5, 2.5 s in a 2.5-minute run. The learning lasts only for one run, because a new process starts with a
fresh brain.

**Cost**:
- The loop runs every 0.52 s on average. The brain uses about 0.2 s of one CPU core per loop and 150-425 MB of
  RAM.
- The sandbox server uses about 3% CPU and 470 MB of RAM, with a hard cap at 1 GB.

## Safety: the sandbox

- **One server only.** The fly administers only the ClickHouse server that `cluster.py` starts. It runs the
  ClickHouse binary from `CLICKHOUSE_BINARY` (or `clickhouse` on the PATH). The data lives in `ch-data/`.
- **Local only.** The server listens on 127.0.0.1 only, on HTTP port 8161 and TCP port 9161.
- **Checked before use.** Before touching anything, `start()` checks that the server on port 8161 reports
  `ch-data/` as its path. If it does not, it refuses. HTTP requests skip any proxy.
- **Locked-down config.** `ch-data/config.xml` turns off the MySQL and PostgreSQL ports and crash reports,
  accepts the default user only from 127.0.0.1, caps server memory at 1 GB and uses 2 threads per query.
- **Fixed action list.** The fly can only do the four actions in the table above, on the `fly_demo` database. It
  cannot drop anything, delete user data or change settings.
- **What chaos does.** Chaos only inserts small batches of rows, stops merges on a `fly_demo` table, runs a
  `sleepEachRow` query that uses no CPU, and sends queries that fail.

## Known limits

- **Kenyon cells stay silent.** They are driven by smell, and no sense here is a smell. MBONs fire only during
  the dopamine burst.
- **Walking is never chosen.** No sense drives the walking DNs above about 2 Hz.
- **Credit spills between actions.** Escape rewards also strengthen some inputs that antennal touch uses. When
  touch comes on, the escape drive can rise to 5-10 Hz. That leads to an occasional pointless `escape` during a
  too-many-parts issue (2 of 39 actions in the run above).
- **The fly can "feed" at the wrong moment.** When a parts issue has just opened, leftover sugar activity can make
  it choose `feed` once. This is harmless.
- **No forgetting.** Weights never decay, so a long run ends with the rewarded pathways at the 3x cap.
- **Slow brain.** The brain runs slower than real time: each 0.5 s loop simulates 200 ms of brain time.

## Files

- `run.py`: the main loop that wires everything together.
- `brain.py`: `FlyBrain`, the whole-brain leaky integrate-and-fire model with its groups and learning.
  `brain.py --prep` converts the data.
- `cluster.py`: `Sandbox`, which starts and stops the server, injects chaos, reads metrics and senses, and runs
  the actions.
- `ui/server.py` and `ui/index.html`: the live page, sent to the browser as Server-Sent Events.
- `validate_brain.py`: the brain checks, including sugar driving feeding and bitter not.
- `data/`: the connectome parquet files and the compact numpy CSR arrays. `groups.json` lists the annotation each
  neuron group comes from.
- `runs/`: recorded runs.

## Data and credits

- **Connectome**: FlyWire FAFB v783, via the Hugging Face dataset
  [SLOP011/flywire-fafb-connectome](https://huggingface.co/datasets/SLOP011/flywire-fafb-connectome)
  (`connections.parquet`, `nodes.parquet`; CC-BY-4.0). It repackages the FlyWire v783 connectivity release
  (Zenodo, doi:10.5281/zenodo.10676866) and the Schlegel et al. annotations.
- **Model**: the leaky integrate-and-fire parameters and the sugar-to-MN9 check follow Shiu et al.
- **Papers**:
  - Dorkenwald, S. et al. *Neuronal wiring diagram of an adult brain.* Nature 634 (2024).
    doi:10.1038/s41586-024-07558-y
  - Schlegel, P. et al. *Whole-brain annotation and multi-connectome cell typing of Drosophila.* Nature 634
    (2024). doi:10.1038/s41586-024-07686-5
  - Shiu, P. K. et al. *A Drosophila computational brain model reveals sensorimotor processing.* Nature 634
    (2024). doi:10.1038/s41586-024-07763-9
- This project is not affiliated with the FlyWire Consortium. Explore the real data at
  [codex.flywire.ai](https://codex.flywire.ai).
