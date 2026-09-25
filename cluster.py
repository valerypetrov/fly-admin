"""Fly Admin sandbox: a throwaway local ClickHouse that the fly administers.

Only the server started here, on 127.0.0.1:8161 (HTTP) and 9161 (TCP), is ever contacted.
"""
import json
import os
import random
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque

# The ClickHouse binary: $CLICKHOUSE_BINARY, else `clickhouse` from PATH.
BINARY = os.environ.get('CLICKHOUSE_BINARY') or shutil.which('clickhouse') or 'clickhouse'
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ch-data')
PID_FILE = os.path.join(DATA, 'clickhouse.pid')
TCP_PORT, HTTP_PORT = 9161, 8161
URL = f'http://127.0.0.1:{HTTP_PORT}/'
DB = 'fly_demo'
CHAOS = 'fly_chaos_'

# The fixed whitelist of what the fly can do to the sandbox.
ACTIONS = {
    'groom': 'SYSTEM START MERGES + OPTIMIZE TABLE ... FINAL on the fly_demo table with the most parts',
    'escape': "KILL QUERY for queries whose query_id starts with 'fly_chaos_'",
    'feed': 'do nothing, all is good',
    'walk': 'SYSTEM DROP MARK CACHE (harmless)',
}
KINDS = ('too_many_parts', 'slow_query', 'errors')
SENSE_OF = {'too_many_parts': 'antennal_mechano', 'slow_query': 'looming', 'errors': 'bitter'}

PARTS_OPEN = 10       # this many active parts in one table is too many
SLOW_OPEN_S = 2.0     # a chaos query running this long is slow
ERRORS_OPEN = 0.5     # this many errors per second is an error burst
WINDOW_S = 10.0       # window for error rate and queries per second

# Demo tables: columns + engine, and a SELECT that makes n rows for them.
TABLES = {
    'events': ('(ts DateTime, user_id UInt32, kind LowCardinality(String), value Float64) '
               'ENGINE = MergeTree ORDER BY (kind, ts)',
               "SELECT now() - number, rand() % 1000, ['click', 'view', 'buy'][rand() % 3 + 1], "
               'rand() / 1e6 FROM numbers({n})'),
    'page_views': ('(ts DateTime, user_id UInt32, url String, ms UInt32) ENGINE = MergeTree ORDER BY ts',
                   "SELECT now() - number, rand() % 1000, concat('/page/', toString(rand() % 50)), "
                   'rand() % 3000 FROM numbers({n})'),
    'users': ('(user_id UInt32, name String, signup Date) ENGINE = MergeTree ORDER BY user_id',
              "SELECT number, concat('fly', toString(number)), today() - rand() % 365 FROM numbers({n})"),
}
BAD_QUERIES = [f'SELECT * FROM {DB}.no_such_table', "SELECT throwIf(1, 'fly chaos')", 'SELEC 1']

CONFIG = f"""<clickhouse>
    <logger><level>warning</level><log>{DATA}/log/server.log</log>
        <errorlog>{DATA}/log/server.err.log</errorlog><size>10M</size><count>1</count></logger>
    <tmp_path>{DATA}/tmp/</tmp_path>
    <user_files_path>{DATA}/user_files/</user_files_path>
    <format_schema_path>{DATA}/format_schemas/</format_schema_path>
    <mlock_executable>false</mlock_executable>
    <send_crash_reports><enabled>false</enabled></send_crash_reports>
    <max_server_memory_usage>1000000000</max_server_memory_usage>
    <mark_cache_size>67108864</mark_cache_size>
    <uncompressed_cache_size>67108864</uncompressed_cache_size>
    <users><default><password/><networks><ip>127.0.0.1</ip></networks>
        <profile>default</profile><quota>default</quota></default></users>
    <profiles><default><max_threads>2</max_threads></default></profiles>
    <quotas><default/></quotas>
    <user_directories><users_xml><path>{DATA}/config.xml</path></users_xml></user_directories>
</clickhouse>
"""

METRICS_SQL = f"""SELECT
    (SELECT groupArray((table, n)) FROM (SELECT table, count() AS n FROM system.parts
        WHERE database = '{DB}' AND active GROUP BY table)) AS parts,
    (SELECT count() FROM system.processes WHERE query_id != queryID()) AS queries_running,
    (SELECT max(elapsed) FROM system.processes WHERE query_id != queryID()) AS longest_query_s,
    (SELECT max(elapsed) FROM system.processes WHERE startsWith(query_id, '{CHAOS}')) AS chaos_query_s,
    (SELECT sum(value) FROM system.errors) AS errors_total,
    (SELECT sum(value) FROM system.events WHERE event = 'Query') AS queries_total,
    (SELECT sum(value) FROM system.metrics WHERE metric = 'MemoryTracking') AS memory_bytes,
    (SELECT sum(value) FROM system.metrics WHERE metric = 'Merge') AS merges_running
FORMAT JSONEachRow"""

# No proxies: requests must go straight to the local sandbox.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def query(sql, query_id=None, timeout=10.0):
    """Run one statement on the sandbox over HTTP and return the response text."""
    url = URL + (f'?query_id={query_id}' if query_id else '')
    try:
        with _opener.open(urllib.request.Request(url, data=sql.encode()), timeout=timeout) as r:
            return r.read().decode()
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode(errors='replace').strip()) from None


def severity(m):
    """How bad each issue kind is right now, 0 (fine) to 1 (worst)."""
    parts, slow, errs = m['parts_max'], m['chaos_query_s'], m['error_rate']
    return {
        'too_many_parts': min(1.0, parts / 50) if parts >= PARTS_OPEN else 0.0,
        'slow_query': min(1.0, slow / 60) if slow >= SLOW_OPEN_S else 0.0,
        'errors': min(1.0, errs / 10) if errs >= ERRORS_OPEN else 0.0,
    }


class Sandbox:
    ACTIONS = ACTIONS

    def __init__(self):
        self._lock = threading.Lock()
        self._samples = deque()   # (t, errors_total, queries_total)
        self._open = {}           # kind -> open issue
        self._next_id = 1
        self._stop = threading.Event()
        self._proc = None
        self._injectors = {}      # kind -> chaos thread
        self._chaos = None

    # --- server lifecycle ---

    def start(self):
        """Start the sandbox server if it is not up, then make sure fly_demo exists."""
        if not self._alive():
            os.makedirs(os.path.join(DATA, 'log'), exist_ok=True)
            with open(os.path.join(DATA, 'config.xml'), 'w') as f:
                f.write(CONFIG)
            cmd = [BINARY, 'server', f'--config-file={DATA}/config.xml', f'--pid-file={PID_FILE}', '--',
                   f'--path={DATA}/', f'--tcp_port={TCP_PORT}', f'--http_port={HTTP_PORT}',
                   '--listen_host=127.0.0.1']
            with open(os.path.join(DATA, 'log', 'stdout.log'), 'a') as log:
                self._proc = subprocess.Popen(cmd, cwd=DATA, stdin=subprocess.DEVNULL, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.time() + 90
            while not self._alive():
                if self._proc.poll() is not None or time.time() > deadline:
                    raise RuntimeError(f'sandbox server did not start, see {DATA}/log')
                time.sleep(0.3)
        path = query("SELECT value FROM system.server_settings WHERE name = 'path'").strip()
        if os.path.realpath(path) != os.path.realpath(DATA):
            raise RuntimeError(f'port {HTTP_PORT} is not our sandbox (path {path!r})')
        query(f'CREATE DATABASE IF NOT EXISTS {DB}')
        for name, (columns, rows) in TABLES.items():
            query(f'CREATE TABLE IF NOT EXISTS {DB}.{name} {columns}')
            if query(f'SELECT count() FROM {DB}.{name}').strip() == '0':
                query(f'INSERT INTO {DB}.{name} ' + rows.format(n=10000))
        self._stop.clear()
        return self

    def stop(self):
        """Stop chaos and the sandbox server. Safe to call twice."""
        self._stop.set()
        try:
            query(f"KILL QUERY WHERE startsWith(query_id, '{CHAOS}') SYNC", timeout=15)
        except Exception:
            pass
        pid = self._pid()
        if pid is None and self._proc is not None and self._proc.poll() is None:
            pid = self._proc.pid  # stopped while starting: no pid file yet, so use the child we started
        if pid is None:
            return
        os.kill(pid, signal.SIGTERM)
        if not self._wait_gone(pid, 30):
            os.kill(pid, signal.SIGKILL)
            self._wait_gone(pid, 10)

    def _alive(self):
        try:
            with _opener.open(URL + 'ping', timeout=1) as r:
                return r.read().strip() == b'Ok.'
        except OSError:
            return False

    def _pid(self):
        """The server pid from our pid file, only if that process is our binary."""
        try:
            pid = int(open(PID_FILE).read().strip())
            cmd = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout
        except (OSError, ValueError):
            return None
        return pid if cmd.startswith(BINARY) else None

    def _wait_gone(self, pid, timeout):
        end = time.time() + timeout
        while time.time() < end:
            if self._proc is not None and self._proc.poll() is not None:
                return True
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.2)
        return False

    # --- chaos ---

    def inject(self, kind):
        """Start one real issue of this kind in a background thread; False if one is already running."""
        running = self._injectors.get(kind)
        if running is not None and running.is_alive():
            return False
        target = {'too_many_parts': self._chaos_parts, 'slow_query': self._chaos_slow,
                  'errors': self._chaos_errors}[kind]
        t = threading.Thread(target=target, daemon=True, name=f'chaos-{kind}')
        self._injectors[kind] = t
        t.start()
        return True

    def chaos(self, every_s=30.0, overlap=0.3):
        """Start a timer that injects a random issue every ~every_s seconds.
        When an issue is already open, a second one is added with probability `overlap`."""
        if self._chaos is not None and self._chaos.is_alive():
            return

        def loop():
            while not self._stop.wait(random.uniform(0.5, 1.5) * every_s):
                try:
                    sev = severity(self.metrics())
                except Exception:
                    continue
                busy = [k for k in KINDS if sev[k] > 0 or
                        (k in self._injectors and self._injectors[k].is_alive())]
                if busy and random.random() > overlap:
                    continue
                free = [k for k in KINDS if k not in busy]
                if free:
                    self.inject(random.choice(free))

        self._chaos = threading.Thread(target=loop, daemon=True, name='chaos')
        self._chaos.start()

    def _chaos_parts(self):
        # Merges stay stopped until the fly grooms; one small part lands every 0.5 s for 30 s.
        table = random.choice(list(TABLES))
        try:
            query(f'SYSTEM STOP MERGES {DB}.{table}')
            for _ in range(60):
                if self._stop.is_set():
                    break
                query(f'INSERT INTO {DB}.{table} ' + TABLES[table][1].format(n=100))
                time.sleep(0.5)
        except Exception:
            pass

    def _chaos_slow(self):
        # Sleeps 1 s per row for up to 5 minutes and uses no CPU; the fly should KILL it.
        qid = f'{CHAOS}slow_{os.urandom(4).hex()}'
        try:
            query('SELECT sleepEachRow(1) FROM numbers(300) SETTINGS max_block_size = 1 FORMAT Null',
                  query_id=qid, timeout=400)
        except Exception:
            pass

    def _chaos_errors(self):
        # A 15-30 s burst of failing queries at 2-10 per second; it heals by itself.
        end, pause = time.time() + random.uniform(15, 30), 1 / random.uniform(2, 10)
        while time.time() < end and not self._stop.is_set():
            try:
                query(random.choice(BAD_QUERIES), query_id=f'{CHAOS}err_{os.urandom(4).hex()}')
            except Exception:
                pass
            time.sleep(pause)

    # --- what the fly sees ---

    def metrics(self):
        """Numbers from system tables; cheap enough to call every 0.5 s."""
        r = json.loads(query(METRICS_SQL))
        now = time.time()
        parts = {table: int(n) for table, n in r['parts']}
        errors, queries = int(r['errors_total']), int(r['queries_total'])
        with self._lock:
            self._samples.append((now, errors, queries))
            while len(self._samples) > 1 and self._samples[1][0] <= now - WINDOW_S:
                self._samples.popleft()
            t0, e0, q0 = self._samples[0]
        dt = now - t0
        return {
            't': now,
            'parts': parts,
            'parts_max': max(parts.values(), default=0),
            'parts_total': sum(parts.values()),
            'queries_running': int(r['queries_running']),
            'longest_query_s': float(r['longest_query_s']),
            'chaos_query_s': float(r['chaos_query_s']),
            'error_rate': max(0, errors - e0) / dt if dt > 0 else 0.0,
            'errors_total': errors,
            'qps': max(0, queries - q0) / dt if dt > 0 else 0.0,
            'memory_bytes': int(r['memory_bytes']),
            'merges_running': int(r['merges_running']),
        }

    def senses(self, m):
        """Poisson rates (Hz) for the brain's sensory groups: an open issue gives 50 Hz rising
        to 200 Hz at full severity; 'sugar' is 150 Hz only while nothing is open."""
        sev = severity(m)
        hz = {SENSE_OF[k]: 50.0 + 150.0 * s if s > 0 else 0.0 for k, s in sev.items()}
        hz['sugar'] = 0.0 if any(sev.values()) else 150.0
        return hz

    def issues(self, m=None):
        """Open issues as [{'id', 'kind', 'since', 'severity'}]; pass the latest metrics() to save a query."""
        m = m or self.metrics()
        sev = severity(m)
        with self._lock:
            for kind, s in sev.items():
                if s > 0 and kind not in self._open:
                    self._open[kind] = {'id': f'{kind}#{self._next_id}', 'kind': kind, 'since': m['t']}
                    self._next_id += 1
                elif s == 0:
                    self._open.pop(kind, None)
            slow = self._open.get('slow_query')
        if slow is not None and 'query' not in slow:
            # Remember the slow query's text, so the UI can show it.
            text = query(f"SELECT query FROM system.processes WHERE startsWith(query_id, '{CHAOS}') "
                         'ORDER BY elapsed DESC LIMIT 1').strip()
            if text:
                slow['query'] = text
        with self._lock:
            return [dict(i, severity=round(sev[i['kind']], 3)) for i in self._open.values()]

    @staticmethod
    def resolved(prev_issues, now_issues):
        """Ids that were open before and are closed now."""
        now_ids = {i['id'] for i in now_issues}
        return [i['id'] for i in prev_issues if i['id'] not in now_ids]

    # --- what the fly does ---

    def act(self, name):
        """Run one whitelisted action; returns {'ok', 'detail', 'sql', 'queries'}."""
        sql = []
        queries = []
        try:
            if name == 'groom':
                table, before = query(f"SELECT table, count() FROM system.parts WHERE database = '{DB}' "
                                      'AND active GROUP BY table ORDER BY count() DESC, table LIMIT 1'
                                      ).split() or ['events', '0']
                sql = [f'SYSTEM START MERGES {DB}.{table}', f'OPTIMIZE TABLE {DB}.{table} FINAL']
                for s in sql:
                    query(s, timeout=120)
                after = query(f"SELECT count() FROM system.parts WHERE database = '{DB}' "
                              f"AND table = '{table}' AND active").strip()
                detail = f'{DB}.{table}: {before} -> {after} parts'
            elif name == 'escape':
                sql = [f"KILL QUERY WHERE startsWith(query_id, '{CHAOS}') SYNC FORMAT JSONEachRow"]
                rows = [json.loads(line) for line in query(sql[0], timeout=60).splitlines()]
                killed = [r['query_id'] for r in rows]
                queries = [r.get('query', '') for r in rows]
                detail = f'killed {len(killed)} chaos queries {killed}'
            elif name == 'feed':
                detail = 'all good, nothing to do'
            elif name == 'walk':
                sql = ['SYSTEM DROP MARK CACHE']
                query(sql[0])
                detail = 'dropped the mark cache'
            else:
                return {'ok': False, 'detail': f'unknown action {name!r}', 'sql': ''}
            return {'ok': True, 'detail': detail, 'sql': '; '.join(sql), 'queries': queries}
        except Exception as e:
            return {'ok': False, 'detail': str(e)[:300], 'sql': '; '.join(sql)}


if __name__ == '__main__':
    # Self-check: create each issue for real, fix it with its action, show metrics before and after.
    sb = Sandbox().start()
    keys = ('parts_max', 'chaos_query_s', 'queries_running', 'error_rate', 'qps', 'memory_bytes')

    def wait(kind, want_open, timeout):
        end = time.time() + timeout
        while time.time() < end:
            m = sb.metrics()
            if (kind in [i['kind'] for i in sb.issues(m)]) == want_open:
                return m
            time.sleep(0.5)
        raise SystemExit(f'FAIL: {kind} did not {"open" if want_open else "close"} within {timeout} s')

    def show(label, m):
        print(f'  {label:7}', {k: round(m[k], 2) for k in keys}, sb.senses(m), sb.issues(m))

    try:
        for kind, action in (('too_many_parts', 'groom'), ('slow_query', 'escape'), ('errors', 'feed')):
            print(kind)
            show('calm', wait(kind, False, 60))
            sb.inject(kind)
            wait(kind, True, 30)
            time.sleep(5)
            m = sb.metrics()
            show('sick', m)
            prev, t0 = sb.issues(m), time.time()
            print('  act    ', action, sb.act(action))
            m = wait(kind, False, 90)
            show('healed', m)
            print('  resolved', sb.resolved(prev, sb.issues(m)), f'{time.time() - t0:.1f} s after the action')
        print('walk', sb.act('walk'))
    finally:
        sb.stop()
