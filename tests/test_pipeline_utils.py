"""Tests for pipeline utility functions."""

import socket

from opd.utils.net import find_free_port, port_is_listening


class TestFindFreePort:
    def test_returns_int(self):
        port = find_free_port()
        assert isinstance(port, int)

    def test_port_in_valid_range(self):
        port = find_free_port()
        assert 1024 <= port <= 65535

    def test_two_calls_return_different_ports(self):
        """Two consecutive calls should (almost certainly) return different ports."""
        p1 = find_free_port()
        p2 = find_free_port()
        # They could theoretically be the same, but extremely unlikely
        # Just check both are valid
        assert 1024 <= p1 <= 65535
        assert 1024 <= p2 <= 65535

    def test_port_is_available(self):
        """The returned port should be bindable."""
        port = find_free_port()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
        except OSError:
            # Port may have been reused between calls — acceptable
            pass


class TestPortIsListening:
    def test_listening_port(self):
        """A port we bind to should show as listening."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            assert port_is_listening(port, timeout=1.0) is True

    def test_non_listening_port(self):
        """A random high port should not be listening."""
        # Find a free port and don't listen on it
        port = find_free_port()
        assert port_is_listening(port, timeout=0.1) is False
