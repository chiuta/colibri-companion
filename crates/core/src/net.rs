//! Evidence about where this program's traffic goes.
//!
//! Two independent sources, so the claim "nothing leaves this computer" can be checked and
//! not just stated:
//!
//! * a log of every connection this program itself opens (the only place it does so is
//!   [`crate::http::send`], which speaks plain HTTP to the server address in the settings);
//! * the operating system's own table of open sockets for this program and for colibri's
//!   processes, read the way `ss`, `netstat` or `lsof` would show it.

use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Mutex;
use std::time::SystemTime;

// ------------------------------------------------------------------ what this program opened

#[derive(Clone, Debug, PartialEq)]
pub struct Outbound {
    pub at: SystemTime,
    pub addr: SocketAddr,
    pub ok: bool,
}

static LOG: Mutex<Vec<Outbound>> = Mutex::new(Vec::new());
const LOG_CAP: usize = 500;

/// Called for every connection this program tries to open.
pub fn record(addr: SocketAddr, ok: bool) {
    if let Ok(mut log) = LOG.lock() {
        if log.len() >= LOG_CAP {
            log.remove(0);
        }
        log.push(Outbound {
            at: SystemTime::now(),
            addr,
            ok,
        });
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LogSummary {
    pub total: usize,
    pub local: usize,
    /// External destinations (IP only, no port) and how often each was contacted.
    pub external: Vec<(String, usize)>,
}

pub fn summarize(log: &[Outbound]) -> LogSummary {
    let mut s = LogSummary {
        total: log.len(),
        ..Default::default()
    };
    let mut ext: BTreeMap<String, usize> = BTreeMap::new();
    for o in log {
        if is_local_ip(o.addr.ip()) {
            s.local += 1;
        } else {
            *ext.entry(o.addr.ip().to_string()).or_default() += 1;
        }
    }
    s.external = ext.into_iter().collect();
    s
}

pub fn log_summary() -> LogSummary {
    LOG.lock().map(|l| summarize(&l)).unwrap_or_default()
}

// ------------------------------------------------------------------ addresses

/// Loopback (this computer talking to itself) and unspecified (a socket that is merely open).
pub fn is_local_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback() || v4.is_unspecified(),
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback())
        }
    }
}

// ------------------------------------------------------------------ the operating system's view

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Waiting for connections (UDP: bound).
    Listening,
    /// Connected, or trying to connect, to a peer.
    Connected,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Conn {
    pub pid: u32,
    pub proto: &'static str,
    pub local: SocketAddr,
    /// `None` for a listening or merely bound socket.
    pub remote: Option<SocketAddr>,
    pub kind: Kind,
}

impl Conn {
    /// Does this socket talk to, or accept from, anything beyond this computer?
    pub fn leaves_computer(&self) -> bool {
        match (self.kind, self.remote) {
            (Kind::Connected, Some(r)) => !is_local_ip(r.ip()),
            _ => false,
        }
    }
    /// Is a listening socket reachable from other computers (not bound to loopback)?
    pub fn exposed_listener(&self) -> bool {
        self.kind == Kind::Listening && !is_loopback_only(self.local.ip()) && self.proto == "TCP"
    }
}

fn is_loopback_only(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback(),
        IpAddr::V6(v6) => v6.is_loopback() || v6.to_ipv4_mapped().is_some_and(|v| v.is_loopback()),
    }
}

/// `host:port`, `[v6]:port` or `*:port`.
pub fn parse_endpoint(text: &str) -> Option<SocketAddr> {
    let text = text.trim();
    if let Some(port) = text.strip_prefix("*:") {
        return Some(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            port.parse().ok()?,
        ));
    }
    if let Ok(a) = text.parse::<SocketAddr>() {
        return Some(a);
    }
    // an IPv6 address with a zone, `fe80::1%eth0:80`, or lsof's `[::1]:8000` forms
    let (host, port) = text.rsplit_once(':')?;
    let host = host.trim_matches(['[', ']']);
    let host = host.split('%').next()?;
    Some(SocketAddr::new(host.parse().ok()?, port.parse().ok()?))
}

/// Windows: the output of `netstat -ano`.
pub fn parse_netstat_windows(text: &str, pids: &[u32]) -> Vec<Conn> {
    let mut out = Vec::new();
    for line in text.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        let (proto, local, remote, state, pid) = match cols.as_slice() {
            ["TCP", l, r, s, p] => ("TCP", l, r, Some(*s), p),
            ["UDP", l, r, p] => ("UDP", l, r, None, p),
            _ => continue,
        };
        let Ok(pid) = pid.parse::<u32>() else {
            continue;
        };
        if !pids.contains(&pid) {
            continue;
        }
        let Some(local) = parse_endpoint(local) else {
            continue;
        };
        let listening = state == Some("LISTENING") || proto == "UDP";
        let remote = if listening {
            None
        } else {
            parse_endpoint(remote)
        };
        if state == Some("TIME_WAIT") {
            continue;
        }
        out.push(Conn {
            pid,
            proto,
            local,
            remote,
            kind: if remote.is_some() {
                Kind::Connected
            } else {
                Kind::Listening
            },
        });
    }
    out
}

/// macOS: the output of `lsof -nP -i -a -p <pids>`.
pub fn parse_lsof(text: &str) -> Vec<Conn> {
    let mut out = Vec::new();
    for line in text.lines().skip(1) {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 9 {
            continue;
        }
        let Ok(pid) = cols[1].parse::<u32>() else {
            continue;
        };
        let proto = match cols.iter().position(|c| *c == "TCP" || *c == "UDP") {
            Some(i) if cols[i] == "TCP" => "TCP",
            Some(_) => "UDP",
            None => continue,
        };
        let i = cols.iter().position(|c| *c == proto).unwrap_or(0);
        let Some(name) = cols.get(i + 1) else {
            continue;
        };
        let (local_s, remote_s) = match name.split_once("->") {
            Some((l, r)) => (l, Some(r)),
            None => (*name, None),
        };
        let Some(local) = parse_endpoint(local_s) else {
            continue;
        };
        let remote = remote_s.and_then(parse_endpoint);
        out.push(Conn {
            pid,
            proto,
            local,
            remote,
            kind: if remote.is_some() {
                Kind::Connected
            } else {
                Kind::Listening
            },
        });
    }
    out
}

/// Linux: one `/proc/net/{tcp,tcp6,udp,udp6}` table, keeping the sockets in `inodes`.
/// Returns `(inode, conn-without-pid)`.
pub fn parse_proc_net(text: &str, proto: &'static str, v6: bool) -> Vec<(u64, Conn)> {
    let mut out = Vec::new();
    for line in text.lines().skip(1) {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 10 {
            continue;
        }
        let (Some(local), Some(remote)) = (hex_endpoint(cols[1], v6), hex_endpoint(cols[2], v6))
        else {
            continue;
        };
        let state = cols[3];
        let Ok(inode) = cols[9].parse::<u64>() else {
            continue;
        };
        if inode == 0 || state == "06" {
            continue; // not a live socket, or TIME_WAIT
        }
        let listening = if proto == "TCP" {
            state == "0A"
        } else {
            remote.ip().is_unspecified()
        };
        out.push((
            inode,
            Conn {
                pid: 0,
                proto,
                local,
                remote: if listening { None } else { Some(remote) },
                kind: if listening {
                    Kind::Listening
                } else {
                    Kind::Connected
                },
            },
        ));
    }
    out
}

fn hex_endpoint(text: &str, v6: bool) -> Option<SocketAddr> {
    let (addr, port) = text.split_once(':')?;
    let port = u16::from_str_radix(port, 16).ok()?;
    let ip = if v6 {
        if addr.len() != 32 {
            return None;
        }
        let mut b = [0u8; 16];
        for w in 0..4 {
            // each 32-bit word is written in the machine's (little-endian) byte order
            let word = u32::from_str_radix(&addr[w * 8..w * 8 + 8], 16).ok()?;
            b[w * 4..w * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        IpAddr::V6(Ipv6Addr::from(b))
    } else {
        if addr.len() != 8 {
            return None;
        }
        IpAddr::V4(Ipv4Addr::from(
            u32::from_str_radix(addr, 16).ok()?.to_le_bytes(),
        ))
    };
    Some(SocketAddr::new(ip, port))
}

/// The open sockets of `pids`, as the operating system reports them right now.
pub fn connections_of(pids: &[u32]) -> Result<Vec<Conn>, String> {
    if pids.is_empty() {
        return Ok(Vec::new());
    }
    #[cfg(target_os = "linux")]
    {
        linux_connections(pids)
    }
    #[cfg(windows)]
    {
        let mut c = std::process::Command::new("netstat");
        c.arg("-ano");
        crate::proc::hide_window(&mut c);
        let out = crate::proc::run_capture(c, std::time::Duration::from_secs(15))
            .map_err(|e| e.to_string())?;
        Ok(parse_netstat_windows(
            &String::from_utf8_lossy(&out.stdout),
            pids,
        ))
    }
    #[cfg(target_os = "macos")]
    {
        let list: Vec<String> = pids.iter().map(u32::to_string).collect();
        let mut c = std::process::Command::new("lsof");
        c.args(["-nP", "-i", "-a", "-p", &list.join(",")]);
        let out = crate::proc::run_capture(c, std::time::Duration::from_secs(15))
            .map_err(|e| e.to_string())?;
        // lsof exits with 1 when nothing matches; that is a valid, empty answer
        Ok(parse_lsof(&String::from_utf8_lossy(&out.stdout)))
    }
    #[cfg(not(any(target_os = "linux", windows, target_os = "macos")))]
    {
        Err("not supported on this system".into())
    }
}

#[cfg(target_os = "linux")]
fn linux_connections(pids: &[u32]) -> Result<Vec<Conn>, String> {
    use std::collections::HashMap;
    let mut owner: HashMap<u64, u32> = HashMap::new();
    for &pid in pids {
        let Ok(rd) = std::fs::read_dir(format!("/proc/{pid}/fd")) else {
            continue;
        };
        for e in rd.flatten() {
            if let Ok(target) = std::fs::read_link(e.path()) {
                let t = target.to_string_lossy();
                if let Some(n) = t.strip_prefix("socket:[").and_then(|s| s.strip_suffix(']')) {
                    if let Ok(inode) = n.parse::<u64>() {
                        owner.insert(inode, pid);
                    }
                }
            }
        }
    }
    let mut out = Vec::new();
    for (file, proto, v6) in [
        ("/proc/net/tcp", "TCP", false),
        ("/proc/net/tcp6", "TCP", true),
        ("/proc/net/udp", "UDP", false),
        ("/proc/net/udp6", "UDP", true),
    ] {
        let Ok(text) = std::fs::read_to_string(file) else {
            continue;
        };
        for (inode, mut c) in parse_proc_net(&text, proto, v6) {
            if let Some(&pid) = owner.get(&inode) {
                c.pid = pid;
                out.push(c);
            }
        }
    }
    Ok(out)
}

/// The command a person can run themselves to see the same thing.
pub fn verify_command() -> &'static str {
    if cfg!(windows) {
        "netstat -ano"
    } else if cfg!(target_os = "macos") {
        "lsof -nP -i"
    } else {
        "ss -tunap"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_and_external_addresses() {
        assert!(is_local_ip("127.0.0.1".parse().unwrap()));
        assert!(is_local_ip("::1".parse().unwrap()));
        assert!(is_local_ip("::ffff:127.0.0.1".parse().unwrap()));
        assert!(!is_local_ip("140.82.112.3".parse().unwrap()));
        assert!(!is_local_ip("192.168.1.20".parse().unwrap()));
    }

    #[test]
    fn the_app_log_is_summarised() {
        let at = SystemTime::now();
        let mk = |a: &str| Outbound {
            at,
            addr: a.parse().unwrap(),
            ok: true,
        };
        let s = summarize(&[
            mk("127.0.0.1:8000"),
            mk("127.0.0.1:8000"),
            mk("10.0.0.5:8000"),
        ]);
        assert_eq!((s.total, s.local), (3, 2));
        assert_eq!(s.external, vec![("10.0.0.5".to_string(), 1)]);
    }

    #[test]
    fn reads_proc_net_tcp() {
        let text = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n\
   0: 0100007F:1F40 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 11111 1 0\n\
   1: 0100007F:C350 0100007F:1F40 01 00000000:00000000 00:00000000 00000000  1000        0 22222 1 0\n\
   2: 0A00000A:D2F6 8C70520A:01BB 01 00000000:00000000 00:00000000 00000000  1000        0 33333 1 0\n\
   3: 0100007F:C351 0100007F:1F40 06 00000000:00000000 00:00000000 00000000  1000        0 0 1 0\n";
        let rows = parse_proc_net(text, "TCP", false);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].1.kind, Kind::Listening);
        assert_eq!(rows[0].1.local, "127.0.0.1:8000".parse().unwrap());
        assert!(!rows[0].1.exposed_listener());
        assert!(!rows[1].1.leaves_computer());
        assert_eq!(rows[2].1.remote, Some("10.82.112.140:443".parse().unwrap()));
        assert!(rows[2].1.leaves_computer());
    }

    #[test]
    fn reads_ipv6_and_listeners_open_to_the_network() {
        let text = "  sl  local_address                         remote_address                        st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n\
   0: 00000000000000000000000001000000:1F40 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 44444 1 0\n\
   1: 00000000000000000000000000000000:1F41 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 55555 1 0\n";
        let rows = parse_proc_net(text, "TCP", true);
        assert_eq!(rows[0].1.local, "[::1]:8000".parse().unwrap());
        assert!(!rows[0].1.exposed_listener());
        assert!(rows[1].1.exposed_listener());
    }

    #[test]
    fn reads_windows_netstat() {
        let text = "  Proto  Local Address          Foreign Address        State           PID\n\
  TCP    127.0.0.1:8000         0.0.0.0:0              LISTENING       4242\n\
  TCP    127.0.0.1:50321        127.0.0.1:8000         ESTABLISHED     5151\n\
  TCP    192.168.1.5:50400      140.82.112.3:443       ESTABLISHED     4242\n\
  TCP    [::1]:8000             [::]:0                 LISTENING       4242\n\
  TCP    192.168.1.5:50401      1.2.3.4:443            ESTABLISHED     999\n\
  UDP    0.0.0.0:5353           *:*                                    4242\n";
        let c = parse_netstat_windows(text, &[4242, 5151]);
        assert_eq!(c.len(), 5);
        assert_eq!(c.iter().filter(|c| c.leaves_computer()).count(), 1);
        assert!(
            c.iter()
                .any(|c| c.proto == "UDP" && c.kind == Kind::Listening)
        );
    }

    #[test]
    fn reads_lsof() {
        let text = "COMMAND  PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME\n\
python3 4242 me    5u  IPv4 0xabc      0t0  TCP 127.0.0.1:8000 (LISTEN)\n\
python3 4242 me    6u  IPv4 0xabd      0t0  TCP 127.0.0.1:8000->127.0.0.1:50321 (ESTABLISHED)\n\
python3 4242 me    7u  IPv4 0xabe      0t0  TCP 10.0.0.5:50400->140.82.112.3:443 (ESTABLISHED)\n\
python3 4242 me    8u  IPv6 0xabf      0t0  TCP [::1]:9000 (LISTEN)\n";
        let c = parse_lsof(text);
        assert_eq!(c.len(), 4);
        assert_eq!(c.iter().filter(|c| c.leaves_computer()).count(), 1);
        assert_eq!(c[3].local, "[::1]:9000".parse().unwrap());
    }

    /// The real thing: this very test process has a socket we open and can find.
    #[cfg(target_os = "linux")]
    #[test]
    fn finds_a_socket_this_process_owns() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let conns = connections_of(&[std::process::id()]).unwrap();
        assert!(
            conns.iter().any(|c| c.kind == Kind::Listening
                && c.local.port() == port
                && !c.exposed_listener()),
            "{conns:?}"
        );
    }
}
