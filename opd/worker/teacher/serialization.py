"""Tensor-efficient serialization for ZMQ transport."""

import io

import torch


def serialize(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def deserialize(data):
    buf = io.BytesIO(data)
    return torch.load(buf, weights_only=True)
