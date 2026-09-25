#!/usr/bin/env python3
"""Fly Admin: the FlyWire fly brain senses a local ClickHouse sandbox, acts on it, and learns from dopamine.

Run: .venv/bin/python run.py [--minutes N] [--record [PATH]]   then open http://127.0.0.1:8765/
or, from another machine on the network, http://<this machine>:8765/ (the address is printed at start).
"""
import argparse
import json
import math
import os
import signal
import threading
import time

from brain import ACTION_TO_GROUP, FlyBrain
from cluster import KINDS, Sandbox
from ui.server import lan_urls, start_ui

HERE = os.path.dirname(os.path.abspath(__file__))
TICK_S = 0.5          # wall seconds per loop
WINDOW_MS = 200.0     # brain time simulated per loop
ACT_HZ = 8.0          # a motor drive must beat its resting baseline by this much to act
MARGIN_HZ = 3.0       # and beat the runner-up drive by this much
BIAS_HZ = 10.0        # drive added per unit of learned input (1 = the group's original input again)
COOLDOWN_S = 2.0      # pause after any action
REPEAT_S = 10.0       # pause before the same action again, unless a new issue opened since
CHAOS_EVERY_S = 6.0   # mean gap between injected issues
ERRORS_DA = 0.3       # dopamine for staying calm through an error burst
FIXES = {'groom': 'too_many_parts', 'escape': 'slow_query'}  # used only to spot pointless actions


def dopamine_for(ttf):
    # A fast fix gives about 1.0, a 30 s fix about 0.37, never below 0.2.
    return max(0.2, math.exp(-ttf / 30.0))


def run(minutes, record, host):
    ui = start_ui(8765, host)
    print('UI:', ui.url, flush=True)
    if host == '0.0.0.0':
        print('UI from another machine:', ' or '.join(lan_urls(8765)), flush=True)
    rec = open(record, 'a') if record else None

    lock = threading.Lock()  # pokes publish from the web server's thread

    def publish(ev):
        with lock:
            ui.publish(ev)
            if rec:
                rec.write(json.dumps(ev, default=lambda o: o.tolist()) + '\n')
                rec.flush()

    sb = Sandbox()
    try:
        sb.start()
        print('sandbox: ClickHouse on 127.0.0.1:8161 (HTTP) / 9161 (TCP), database fly_demo', flush=True)
        brain = FlyBrain()
        info = brain.info()
        print(f"brain: {info['neurons']:,} neurons, {info['synapses']:,} synapses "
              f"({info['connections']:,} connections)", flush=True)
        publish({'type': 'hello', 'neurons': info['neurons'], 'synapses': info['synapses'],
                 'groups': info['groups'], 'raster_groups': brain.raster_slots})

        # Resting drive with no stimulus, averaged over a few windows.
        rest = [brain.step({}, WINDOW_MS) for _ in range(3)]
        baseline = {a: sum(r[g] for r in rest) / len(rest) for a, g in ACTION_TO_GROUP.items()}

        sb.chaos(every_s=CHAOS_EVERY_S, overlap=0.5)
        t0 = time.time()

        def poke(kind, who):
            # Someone on the network makes trouble on purpose.
            if kind not in KINDS:
                return False, 'unknown trouble'
            if not sb.inject(kind):
                return False, 'already happening'
            who = who.removeprefix('::ffff:')  # show IPv4 callers as plain IPv4
            publish({'type': 'poke', 't': round(time.time() - t0, 2), 'kind': kind, 'by': who})
            return True, 'started'

        ui.on_poke = poke
        end = t0 + minutes * 60 if minutes else math.inf
        prev, history, last = [], [], {}  # history: (time, action, pointless)
        while time.time() < end:
            started = time.time()
            m = sb.metrics()
            issues = sb.issues(m)
            senses = sb.senses(m)
            rates = brain.step(senses, WINDOW_MS)
            t = round(m['t'] - t0, 2)

            # One dopamine burst per issue that closed since the last tick.
            for iid in sb.resolved(prev, issues):
                issue = next(i for i in prev if i['id'] == iid)
                ttf = m['t'] - issue['since']
                during = [(a, p) for ts, a, p in history if ts >= issue['since']]
                if issue['kind'] == 'errors':  # heals by itself; calm pays only if nothing pointless was done
                    chosen, da = (None, 0.0) if any(p for _, p in during) else ('feed', ERRORS_DA)
                else:  # credit the last action taken while it was open
                    chosen, da = (during[-1][0], dopamine_for(ttf)) if during else (None, 0.0)
                if chosen:
                    brain.reward(da, chosen=chosen)
                    publish({'type': 'reward', 't': t, 'fixed': [iid], 'kind': issue['kind'], 'action': chosen,
                             'dopamine': round(da, 3), 'time_to_fix': round(ttf, 2)})
                    print(f'{t:7.1f}s  reward  {iid} fixed in {ttf:.1f} s by {chosen}, dopamine {da:.2f}', flush=True)

            # Drive per action: its motor group's rate plus the learned bias for what the fly senses now.
            bias = brain.action_bias()
            drive = {a: round(rates[g] + BIAS_HZ * bias.get(g, 0.0), 2) for a, g in ACTION_TO_GROUP.items()}
            best, runner = sorted(drive, key=drive.get, reverse=True)[:2]
            snap = brain.snapshot()
            publish({'type': 'tick', 't': t, 'metrics': {k: v for k, v in m.items() if k != 't'},
                     'issues': issues, 'senses': senses, 'brain': snap['groups'], 'raster': snap['raster'],
                     'dopamine': snap['dopamine'], 'learned': snap['learned']})

            now = time.time()
            prior = last.get(best, -math.inf)
            fresh = any(i['since'] > prior for i in issues)
            ready = now - max(last.values(), default=-math.inf) >= COOLDOWN_S and \
                (fresh or now - prior >= REPEAT_S)
            if ready and drive[best] - baseline[best] >= ACT_HZ and drive[best] - drive[runner] >= MARGIN_HZ:
                pointless = best != 'feed' and FIXES.get(best) not in {i['kind'] for i in issues}
                result = sb.act(best)
                last[best] = now
                history.append((now, best, pointless))
                publish({'type': 'action', 't': t, 'action': best, 'drive': drive, 'result': result})
                print(f"{t:7.1f}s  {best:6s}  drive {drive[best]:.1f} Hz vs {runner} {drive[runner]:.1f} Hz: "
                      f"{result['detail']}", flush=True)
            prev = issues
            time.sleep(max(0.0, TICK_S - (time.time() - started)))
    except KeyboardInterrupt:
        print('stopping...', flush=True)
    finally:
        sb.stop()
        ui.stop()
        if rec:
            rec.close()
            print('recorded to', record, flush=True)


def main():
    ap = argparse.ArgumentParser(description='Fly Admin: a fly brain administers a ClickHouse sandbox')
    ap.add_argument('--minutes', type=float, default=0, help='stop after N minutes (default: until Ctrl+C)')
    ap.add_argument('--record', nargs='?', const='', metavar='PATH',
                    help='write every published event as JSON lines (default path: runs/<timestamp>.jsonl)')
    ap.add_argument('--host', default='0.0.0.0', help='UI address: 0.0.0.0 = reachable from the network, 127.0.0.1 = this machine only')
    args = ap.parse_args()
    record = args.record
    if record == '':
        record = os.path.join(HERE, 'runs', time.strftime('%Y%m%d-%H%M%S') + '.jsonl')
    if record:
        os.makedirs(os.path.dirname(os.path.abspath(record)), exist_ok=True)
    for sig in (signal.SIGINT, signal.SIGTERM):  # Ctrl+C and kill both stop cleanly, even when started in the background
        signal.signal(sig, signal.default_int_handler)
    run(args.minutes, record, args.host)


if __name__ == '__main__':
    main()
