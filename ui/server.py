#!/usr/bin/env python3
"""Fly Admin live UI: start_ui(port) serves index.html and an SSE stream at /events, on all interfaces by default.
Run `python3 ui/server.py --demo` for realistic fake events."""
import argparse
import json
import math
import os
import queue
import random
import socket
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


def _plain(o):
    # Make an event JSON-safe: numpy values become Python, NaN and inf become null.
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if hasattr(o, 'tolist'):
        return _plain(o.tolist())
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _sse(event):
    return 'data: ' + json.dumps(event, separators=(',', ':')) + '\n\n'


class Publisher:
    """Broadcasts events to every connected browser. publish() is thread-safe."""

    def __init__(self, server):
        self._server = server
        self._lock = threading.Lock()
        self._clients = set()
        self._hello = None
        self._ticks = deque(maxlen=240)   # (t, data) for the last ~2 minutes of ticks, rasters dropped
        self._events = deque(maxlen=400)  # (t, data) for recent actions and rewards
        self._t = 0.0
        self.url = 'http://127.0.0.1:%d/' % server.server_address[1]

    def publish(self, event):
        event = _plain(event)
        data = _sse(event)
        with self._lock:
            t = event.get('t')
            self._t = t if isinstance(t, (int, float)) and not isinstance(t, bool) else self._t
            if event.get('type') == 'hello':
                self._hello = data
            elif event.get('type') == 'tick':
                self._ticks.append((self._t, _sse(dict(event, raster=[]))))
            else:
                self._events.append((self._t, data))
            for q in self._clients:
                try:
                    q.put_nowait(data)
                except queue.Full:
                    pass  # a stalled browser skips events

    def stop(self):
        with self._lock:
            for q in self._clients:
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass
        self._server.shutdown()
        self._server.server_close()

    def _subscribe(self):
        q = queue.Queue(maxsize=1000)
        with self._lock:
            self._clients.add(q)
            history = sorted(list(self._ticks) + list(self._events), key=lambda e: e[0])  # stable: tick before its action
            backlog = ([self._hello] if self._hello else []) + [d for _, d in history]
        return q, backlog

    def _unsubscribe(self, q):
        with self._lock:
            self._clients.discard(q)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep the console quiet

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path == '/events':
            self._stream()
        elif path in ('/', '/index.html'):
            with open(os.path.join(HERE, 'index.html'), 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def _stream(self):
        pub = self.server.publisher
        q, backlog = pub._subscribe()
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            # History is sent as 'replay' events so the page redraws it without animations.
            self.wfile.write(('retry: 2000\n\n' + ''.join('event: replay\n' + d for d in backlog)).encode())
            self.wfile.flush()
            while True:
                try:
                    data = q.get(timeout=15)
                except queue.Empty:
                    data = ': ping\n\n'
                if data is None:
                    break
                self.wfile.write(data.encode())
                self.wfile.flush()
        except OSError:
            pass  # the browser went away
        finally:
            pub._unsubscribe(q)


def lan_urls(port):
    """URLs that other machines on the network can open."""
    urls = []
    for iface in ('en0', 'en1'):
        try:
            ip = subprocess.run(['ipconfig', 'getifaddr', iface], capture_output=True, text=True, timeout=2).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            ip = ''
        if ip:
            urls.append('http://%s:%d/' % (ip, port))
    urls.append('http://%s:%d/' % (socket.gethostname(), port))
    return urls


class _DualStackServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6

    def server_bind(self):
        # Also accept IPv4, so both the IP address and the .local name (often IPv6) work.
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def start_ui(port=8765, host='0.0.0.0'):
    """Start the UI server on host:port in a background thread; returns a Publisher."""
    if host == '0.0.0.0':
        server = _DualStackServer(('::', port), _Handler)
    else:
        server = ThreadingHTTPServer((host, port), _Handler)
    server.publisher = Publisher(server)
    threading.Thread(target=server.serve_forever, name='fly-ui', daemon=True).start()
    return server.publisher


# ---------------- demo mode: a fake fly and a fake cluster ----------------

SENSES = ['sugar', 'bitter', 'antennal_mechano', 'looming']
ACTIONS = ['groom', 'escape', 'feed', 'walk']
MOTOR = {'groom': 'grooming', 'escape': 'escape', 'feed': 'feeding', 'walk': 'walking'}
KIND_SENSE = {'too_many_parts': 'antennal_mechano', 'slow_query': 'looming', 'errors': 'bitter'}
SIZES = {'sugar': 42, 'bitter': 58, 'antennal_mechano': 470, 'looming': 162, 'kenyon': 5177, 'mbon': 96,
         'dopamine_pam': 307, 'feeding': 2, 'grooming': 6, 'escape': 4, 'walking': 8}
RASTER = [('sugar', 16), ('bitter', 16), ('antennal_mechano', 20), ('looming', 20), ('kenyon', 40), ('mbon', 20),
          ('dopamine_pam', 20), ('feeding', 12), ('grooming', 12), ('escape', 12), ('walking', 12)]
SQL = {'groom': 'OPTIMIZE TABLE fly_demo.{t} FINAL; SYSTEM START MERGES fly_demo.{t}',
       'escape': "KILL QUERY WHERE query_id LIKE 'fly_chaos_%' SYNC",
       'feed': '-- nothing to do', 'walk': 'SYSTEM DROP MARK CACHE'}


def demo(pub, tick=0.5, seed=7):
    rng = random.Random(seed)
    w = {s: {a: round(0.18 + 0.12 * rng.random(), 3) for a in ACTIONS} for s in SENSES}
    w['sugar']['feed'] = 0.7  # innate: sugar drives the proboscis (Shiu et al.)
    for s in KIND_SENSE.values():
        w[s]['feed'] = 0.38  # a naive fly mostly keeps eating when something feels off
    slots, lo = {}, 0
    for g, n in RASTER:
        slots[g] = [lo, lo + n]
        lo += n
    pub.publish({'type': 'hello', 'neurons': 139255, 'synapses': 54492922, 'groups': SIZES,
                 'raster_groups': slots, 'demo': True})
    tables = {'events': 3, 'clicks': 2, 'logs': 4}
    issues, heal_at, acted, nid = [], {}, {}, 0
    next_issue, last_act, da, sim_ms, t0 = 5.0, -9.0, 0.0, 0.0, time.time()
    while True:
        t = time.time() - t0
        free = [k for k in KIND_SENSE if k not in {i['kind'] for i in issues}]
        if t >= next_issue and free:  # at most one open issue per kind
            nid += 1
            kind = rng.choice(free)
            iid = '%s-%d' % (kind, nid)
            issues.append({'id': iid, 'kind': kind, 'since': round(t, 2), 'severity': 0.3})
            heal_at[iid], acted[iid] = t + rng.uniform(7, 12), set()
            next_issue = t + (rng.uniform(2, 5) if rng.random() < 0.25 else rng.uniform(9, 17))
        open_kinds = {i['kind'] for i in issues}
        if 'too_many_parts' in open_kinds:
            tables['events'] = min(300, tables['events'] + rng.randint(4, 9))
        slow = max([t - i['since'] for i in issues if i['kind'] == 'slow_query'] or [0])
        for i in issues:  # a new issue takes ~1.5 s to reach full strength
            raw = {'too_many_parts': max(0.25, tables['events'] / 200), 'slow_query': 0.3 + slow / 20,
                   'errors': 0.4 + 0.3 * rng.random()}[i['kind']]
            i['severity'] = round(min(1.0, (t - i['since']) / 1.5) * min(1.0, raw), 2)
        sev = {k: max([i['severity'] for i in issues if i['kind'] == k] or [0]) for k in KIND_SENSE}
        senses = {'sugar': round(8 + (52 if not issues else 0) + rng.uniform(-4, 4), 1)}
        for k, s in KIND_SENSE.items():
            senses[s] = round(4 + 150 * sev[k] + rng.uniform(-3, 3), 1)
        agitation = 8 + 0.4 * max([t - i['since'] for i in issues] or [0])  # an old issue makes the fly try anything
        drive = {a: round(10 + 40 * sum(senses[s] / 150 * w[s][a] for s in SENSES) + rng.uniform(0, agitation), 1)
                 for a in ACTIONS}
        aroused = max(senses[s] for s in KIND_SENSE.values()) > 30

        # The fly acts every 1.6 s while a sense is loud (6 s when calm); the right action fixes its issue.
        before = {i['id']: i for i in issues}
        choice = max(drive, key=drive.get)
        if t - last_act >= (1.6 if aroused else 6.0):
            last_act = t
            dirty = max(tables, key=tables.get)
            detail = {'groom': 'merged fly_demo.%s: %d -> 1 parts' % (dirty, tables[dirty]),
                      'escape': 'killed %d query' % (1 if 'slow_query' in open_kinds else 0),
                      'feed': 'did nothing harmful', 'walk': 'mark cache dropped'}[choice]
            if choice == 'groom':
                tables[dirty] = 1
            fixes = {'groom': 'too_many_parts', 'escape': 'slow_query'}.get(choice)
            for i in issues:
                acted[i['id']].add(choice)
            issues = [i for i in issues if i['kind'] != fixes]
            pub.publish({'type': 'action', 't': round(t, 2), 'action': choice, 'drive': drive,
                         'result': {'ok': True, 'detail': detail, 'sql': SQL[choice].format(t=dirty)}})
        issues = [i for i in issues if i['kind'] != 'errors' or heal_at[i['id']] > t]
        for iid in sorted(set(before) - {i['id'] for i in issues}):
            kind, ttf = before[iid]['kind'], round(t - before[iid]['since'], 2)
            if kind == 'errors':
                a, dop = 'feed', (0.6 if acted[iid] <= {'feed'} else 0.2)
            else:
                a, dop = choice, round(max(0.4, min(1.0, 1.3 - ttf / 20)), 2)
            da += dop
            sense = KIND_SENSE[kind]
            for b in ACTIONS:  # the sense that saw the issue shifts toward the action that fixed it
                w[sense][b] = round(w[sense][b] + (0.2 if b == a else 0.06) * dop * ((b == a) - w[sense][b]), 3)
            pub.publish({'type': 'reward', 't': round(t, 2), 'fixed': [iid], 'kind': kind,
                         'dopamine': dop, 'time_to_fix': ttf})
        if 'too_many_parts' not in {i['kind'] for i in issues}:
            tables = {k: max(1, v - rng.randint(0, 2)) if v > 4 else v for k, v in tables.items()}

        brain = {g: round(rng.uniform(0.5, 3), 1) for g in SIZES}
        brain.update({s: round(senses[s] * 0.8 + rng.uniform(0, 4), 1) for s in SENSES})
        lo_d, hi_d = min(drive.values()), max(drive.values()) + 1e-9
        brain.update({MOTOR[a]: round(2 + 60 * ((drive[a] - lo_d) / (hi_d - lo_d)) ** 3, 1) for a in ACTIONS})
        brain['kenyon'] = round(1 + 0.02 * sum(senses.values()) + rng.uniform(0, 1), 1)
        brain['mbon'] = round(6 + 0.08 * max(drive.values()) + rng.uniform(0, 3), 1)
        brain['dopamine_pam'] = round(3 + 70 * min(da, 1.2) + rng.uniform(0, 2), 1)
        raster = []
        for g, (first, end) in slots.items():
            for slot in range(first, end):
                n = min(20, max(0, round(rng.gauss(brain[g] * tick, math.sqrt(brain[g] * tick + 0.1)))))
                raster += [[slot, round(sim_ms + rng.uniform(0, tick * 1000), 1)] for _ in range(n)]
        sim_ms += tick * 1000
        metrics = {'max_parts': max(tables.values()), 'parts': dict(tables), 'slow_query_s': round(slow, 1),
                   'processes': 1 + int(slow > 0),
                   'error_rate': round(rng.uniform(15, 30) if sev['errors'] else rng.uniform(0, 0.2), 2),
                   'qps': round(rng.uniform(3, 6) + (9 if 'too_many_parts' in open_kinds else 0), 1),
                   'memory_bytes': int(4.2e8 + 1.5e8 * (slow > 0) + rng.uniform(0, 2e7))}
        pub.publish({'type': 'tick', 't': round(t, 2), 'metrics': metrics, 'issues': issues, 'senses': senses,
                     'brain': brain, 'raster': raster, 'dopamine': round(da, 3), 'learned': w})
        da *= 0.85
        time.sleep(max(0.0, tick - (time.time() - t0 - t)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Fly Admin live UI')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--host', default='0.0.0.0', help='0.0.0.0 = reachable from the network, 127.0.0.1 = this machine only')
    ap.add_argument('--demo', action='store_true', help='stream fake events')
    args = ap.parse_args()
    ui = start_ui(args.port, args.host)
    print('Fly Admin UI on', ui.url, '(demo)' if args.demo else '', flush=True)
    if args.host == '0.0.0.0':
        print('from another machine:', ' or '.join(lan_urls(args.port)), flush=True)
    try:
        demo(ui) if args.demo else threading.Event().wait()
    except KeyboardInterrupt:
        ui.stop()
