"""Task-local pip entry that skips the host's slow IPv6 network path."""
import runpy
import socket

original_getaddrinfo = socket.getaddrinfo


def ipv4_addresses(host, port, family=0, *args, **kwargs):
    if family == 0:
        family = socket.AF_INET
    return original_getaddrinfo(host, port, family, *args, **kwargs)


socket.getaddrinfo = ipv4_addresses
runpy.run_module("pip", run_name="__main__")
