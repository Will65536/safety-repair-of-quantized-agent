"""CPU-only placement-branch/source regression; no PyTorch or model execution.

Exercise the actual added branches with recording stubs, not CUDA devices.
The user-run Slurm quantization job provides the subsequent GPU verification.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

from run_official import ROOT, verify_native_source


class RecordingModule:
    def __init__(self):
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        return self

    def cpu(self):
        self.moves.append("cpu")
        return self


def main():
    verify_native_source()
    for name, function in (("gptq_utils.py", "gptq_fwrd"), ("eval_utils.py", "evaluator")):
        path = ROOT / "fake_quant" / name
        tree = ast.parse(path.read_text())
        target = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        guards = sorted((n for n in ast.walk(target) if isinstance(n, ast.If)
                         and isinstance(n.test, ast.Call) and isinstance(n.test.func, ast.Name)
                         and n.test.func.id == "hasattr" and len(n.test.args) == 2
                         and isinstance(n.test.args[1], ast.Constant) and n.test.args[1].value == "rotary_emb"),
                        key=lambda n: n.lineno)
        assert len(guards) == 2, name
        module = RecordingModule()
        fake_model = SimpleNamespace(model=SimpleNamespace(rotary_emb=module))
        for node in guards:
            compiled = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec")
            exec(compiled, {"model": fake_model, "dev": "cuda:0"})
            # Older Transformers without model-level RoPE must remain supported.
            exec(compiled, {"model": SimpleNamespace(model=SimpleNamespace()), "dev": "cuda:0"})
        assert module.moves == ["cuda:0", "cpu"], (name, module.moves)
        assert fake_model.model.rotary_emb is module
        catcher_line = next(n.lineno for n in ast.walk(target) if isinstance(n, ast.ClassDef) and n.name == "Catcher")
        assert guards[0].lineno < catcher_line < guards[1].lineno
    print("Passed: native numerical-source integrity and GPTQ/evaluator RoPE placement branches (CPU stubs only).")


if __name__ == "__main__":
    main()
