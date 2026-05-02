#!/usr/bin/env python3
"""
DNS-over-HTTPS (DoH) proxy via HTTP proxy.

Resolves DNS queries by sending them as HTTPS requests to DoH servers,
routing through an HTTP proxy (e.g. SSH reverse tunnel on port 7890).

This is needed because the host's DNS (UDP/TCP 53) is blocked by firewall,
but HTTP/HTTPS via proxy works fine.

Usage:
    # Start the proxy (needs sudo for port 53):
    sudo python3 doh_proxy.py --port 53 --proxy http://127.0.0.1:7890

    # Or on high port without sudo:
    python3 doh_proxy.py --port 5353 --proxy http://127.0.0.1:7890

Then update /etc/docker/daemon.json:
    "dns": ["11.0.0.102"]   # host IP where this proxy listens
"""

import argparse
import os
import socket
import struct
import subprocess
import sys
import urllib.request
import urllib.error

DOH_SERVERS = [
    "https://dns.google/dns-query",
    "https://cloudflare-dns.com/dns-query",
    "https://dns.quad9.net/dns-query",
]


def _detect_host_dns():
    try:
        with open("/etc/resolv.conf") as f:
            for line in f:
                line = line.strip()
                if line.startswith("nameserver"):
                    ns = line.split()[1]
                    if ns not in ("127.0.0.53", "127.0.0.1", "0.0.0.0"):
                        return ns
    except Exception:
        pass
    return None


def _detect_proxy():
    for var in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        val = os.environ.get(var)
        if val:
            return val
    for port in (7890, 1080, 8080):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        try:
            s.connect(("127.0.0.1", port))
            s.close()
            return f"http://127.0.0.1:{port}"
        except Exception:
            continue
    return None


def query_doh(dns_query: bytes, doh_url: str, proxy_url: str | None, timeout: float = 5.0) -> bytes | None:
    req = urllib.request.Request(
        doh_url,
        data=dns_query,
        headers={
            "Content-Type": "application/dns-message",
            "Accept": "application/dns-message",
        },
        method="POST",
    )
    opener = None
    if proxy_url:
        proxy_handler = urllib.request.ProxyHandler({
            "http": proxy_url,
            "https": proxy_url,
        })
        opener = urllib.request.build_opener(proxy_handler)

    try:
        if opener:
            resp = opener.open(req, timeout=timeout)
        else:
            resp = urllib.request.urlopen(req, timeout=timeout)
        if resp.status == 200:
            return resp.read()
    except (urllib.error.URLError, OSError) as e:
        pass
    return None


def query_doh_fallback(dns_query: bytes, proxy_url: str | None, timeout: float = 5.0) -> bytes | None:
    for server in DOH_SERVERS:
        result = query_doh(dns_query, server, proxy_url, timeout)
        if result and len(result) > 12:
            return result
    return None


def run_server(listen_addr: str, port: int, proxy_url: str | None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((listen_addr, port))

    host_ip = None
    try:
        result = subprocess.run(
            ["ip", "-4", "route"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "default via" in line and "dev" in line:
                parts = line.split()
                if "src" in parts:
                    host_ip = parts[parts.index("src") + 1]
                break
    except Exception:
        pass

    print(f"DoH DNS proxy listening on {listen_addr}:{port}")
    print(f"Proxy: {proxy_url or 'direct'}")
    print(f"Upstream DoH servers: {DOH_SERVERS}")
    if host_ip:
        print(f"\nDocker DNS config: --dns {host_ip} or daemon.json dns=[\"{host_ip}\"]")
    print()

    while True:
        try:
            data, addr = sock.recvfrom(4096)
            if len(data) < 12:
                continue

            domain = ""
            offset = 12
            while offset < len(data):
                label_len = data[offset]
                if label_len == 0:
                    break
                domain += data[offset + 1 : offset + 1 + label_len].decode("ascii", errors="replace") + "."
                offset += label_len + 1

            print(f"  {domain.rstrip('.')} ", end="", flush=True)

            response = query_doh_fallback(data, proxy_url, timeout=8.0)

            if response:
                rcode = response[3] & 0x0F
                answer_count = struct.unpack("!H", response[6:8])[0]
                print(f"-> OK (ans={answer_count}, rcode={rcode})")
                sock.sendto(response, addr)
            else:
                print("-> FAIL")
                response_fail = bytearray(data)
                response_fail[2] = response_fail[2] | 0x80
                response_fail[3] = response_fail[3] | 0x03
                sock.sendto(bytes(response_fail), addr)
        except KeyboardInterrupt:
            print("\nShutting down.")
            break
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNS-over-HTTPS proxy (via HTTP proxy)")
    parser.add_argument("--listen", default="0.0.0.0", help="Listen address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=53, help="Listen port (default 53)")
    parser.add_argument("--proxy", default=None, help="HTTP proxy URL (e.g. http://127.0.0.1:7890)")
    args = parser.parse_args()

    proxy = args.proxy or _detect_proxy()
    if not proxy:
        print("WARNING: No HTTP proxy detected. DoH queries may fail without DNS.", file=sys.stderr)

    run_server(args.listen, args.port, proxy)
