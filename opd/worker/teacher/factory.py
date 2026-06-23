"""Teacher backend factory — maps backend name to server_main function."""


def get_teacher_backend(name: str) -> dict:
    """Return server_main function for a teacher backend."""
    if name == "vllm":
        from opd.worker.teacher.vllm import teacher_server_main
        return {"server_main": teacher_server_main}
    elif name == "hf":
        from opd.worker.teacher.hf import teacher_hf_server_main
        return {"server_main": teacher_hf_server_main}
    else:
        raise ValueError(f"Unknown teacher backend: '{name}'. Available: vllm, hf")
