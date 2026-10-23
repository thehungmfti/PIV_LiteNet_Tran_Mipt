"""caffe.proto bindings. The shipped caffe_pb2.py was generated with protoc for protobuf 7.x;
if the installed protobuf runtime rejects it, the proto is recompiled with grpcio-tools into a
per-user cache (requires `pip install grpcio-tools`)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load():
    try:
        from piv.models._caffe import caffe_pb2 as mod  # noqa: F401
        return mod
    except Exception as first:  # version mismatch of generated code
        try:
            from grpc_tools import protoc
        except ImportError as e:
            raise ImportError(f"caffe_pb2 import failed ({first}); install grpcio-tools to regenerate") from e
        cache = Path.home() / ".cache" / "piv_caffe_pb2"
        cache.mkdir(parents=True, exist_ok=True)
        rc = protoc.main(["protoc", f"-I{_HERE}", f"--python_out={cache}", str(_HERE / "caffe.proto")])
        if rc != 0:
            raise RuntimeError("protoc failed to compile caffe.proto") from first
        spec = importlib.util.spec_from_file_location("piv_caffe_pb2_regen", cache / "caffe_pb2.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod


caffe_pb2 = _load()
