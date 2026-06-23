"""Async ZMQ client for teacher scoring requests."""

import queue
import threading
import time
from concurrent.futures import Future

import zmq

from opd.worker.teacher.serialization import serialize, deserialize


class TeacherClient:
    """Thread-pool ZMQ client for async teacher requests."""

    def __init__(self, server_addr, n_workers: int = 1):
        # server_addr: single string or list of strings (for teacher DP)
        if isinstance(server_addr, list):
            self.server_addrs = server_addr
            self.n_workers = len(server_addr)
        else:
            self.server_addrs = [server_addr] * n_workers
            self.n_workers = n_workers
        self.server_addr = self.server_addrs[0]  # backward compat
        self.task_queue = queue.Queue()
        self.ctx = zmq.Context()
        self.tracer = None  # set externally for trace spans
        self.teacher_trace_infos = []  # per-worker trace info
        for i in range(self.n_workers):
            threading.Thread(target=self._loop, args=(self.server_addrs[i], i),
                             daemon=True).start()

    def _loop(self, addr=None, worker_idx=0):
        sock = self.ctx.socket(zmq.REQ)
        sock.connect(addr or self.server_addr)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 3600_000)  # 1 hour (long teacher scoring with large n_logprobs)

        while True:
            future, request = self.task_queue.get()
            try:
                t_send = time.monotonic()
                if isinstance(request, dict):
                    payload = dict(request)
                    token_ids = payload.get("prompt_token_ids", [])
                else:
                    token_ids = request
                    payload = {"prompt_token_ids": token_ids}
                data = serialize(payload)
                sock.send(data)
                raw = sock.recv()
                t_recv = time.monotonic()
                t_de = time.monotonic()
                resp = deserialize(raw)
                t_de = time.monotonic() - t_de
                print(f"[TeacherClient] recv={len(raw)/1e6:.1f}MB  "
                      f"deserialize={t_de:.2f}s", flush=True)

                if isinstance(resp, dict) and resp.get("status") == "error":
                    future.set_exception(RuntimeError(resp["reason"]))
                else:
                    timing = resp.get("timing", {})
                    if resp.get("payload_kind") == "hidden_states":
                        teacher_logps = resp.get("teacher_hidden_states", [])
                        teacher_indices = resp.get("teacher_hidden_token_ids", [])
                        teacher_token_logps = resp.get("teacher_hidden_metadata", [])
                    else:
                        teacher_logps = resp.get("teacher_query_logprobs_response")
                        if teacher_logps is None:
                            teacher_logps = resp["teacher_topk_logprobs"]
                        teacher_indices = resp.get("teacher_topk_indices", [])
                        teacher_token_logps = resp.get("teacher_token_logprobs", [])
                    future.set_result((
                        resp["responses"],
                        teacher_logps,
                        teacher_indices,
                        teacher_token_logps,
                        timing.get("mono_start"),
                        timing.get("mono_end"),
                    ))
                    # Emit per-worker trace span
                    tr = self.tracer
                    if tr is not None:
                        info = self.teacher_trace_infos[worker_idx] if worker_idx < len(self.teacher_trace_infos) else {}
                        tr.emit(f"teacher-w{worker_idx}", cat="teacher",
                                tid=20 + worker_idx,  # unique tid per teacher worker
                                t_start=t_send, t_end=t_recv,
                                args={"n_prompts": len(token_ids),
                                      "host": info.get("host", ""),
                                      "gpu_ids": info.get("gpu_ids", "")})
            except Exception as e:
                try:
                    future.set_exception(e)
                except Exception:
                    pass

    def submit(self, prompt_token_ids, prompt_lengths=None, query_indices_response=None,
               query_request_ids=None, teacher_mc_n_total_samples=None,
               return_hidden_states=False, teacher_hidden_dtype=None,
               teacher_hidden_semantics=None):
        """Submit batch of token-id lists. Returns Future[responses, logps, indices]."""
        fut = Future()
        request = prompt_token_ids
        if (prompt_lengths is not None or query_indices_response is not None
                or query_request_ids is not None
                or teacher_mc_n_total_samples is not None
                or return_hidden_states
                or teacher_hidden_dtype is not None
                or teacher_hidden_semantics is not None):
            request = {"prompt_token_ids": prompt_token_ids}
            if prompt_lengths is not None:
                request["prompt_lengths"] = prompt_lengths
            if query_indices_response is not None:
                request["query_indices_response"] = query_indices_response
            if query_request_ids is not None:
                request["query_request_ids"] = query_request_ids
            if teacher_mc_n_total_samples is not None:
                request["teacher_mc_n_total_samples"] = int(teacher_mc_n_total_samples)
            if return_hidden_states:
                request["return_hidden_states"] = True
            if teacher_hidden_dtype is not None:
                request["teacher_hidden_dtype"] = teacher_hidden_dtype
            if teacher_hidden_semantics is not None:
                request["teacher_hidden_semantics"] = teacher_hidden_semantics
        self.task_queue.put((fut, request))
        return fut
