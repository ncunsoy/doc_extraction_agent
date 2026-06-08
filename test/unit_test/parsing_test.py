import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.reformer import ReformerAgent, ReformerOutput
from agents.validator import ValidatorAgent, ValidationResult

# WITHOUT AGENT LOGIC, JUST TEST THE PARSING FUNCTIONS
reformer = ReformerAgent.__new__(ReformerAgent)
validator = ValidatorAgent.__new__(ValidatorAgent)

# REFORMER PARSE
print("[TEST 1] Reformer valid JSON parse")
text = '{"clean_query": "question", "sub_queries": ["a", "b"], "modality": "text", "reasoning": "r"}'
out = reformer._parse(text)
assert out.clean_query == "question", f"Expected 'question', got: {out.clean_query}"
assert out.sub_queries == ["a", "b"], f"Expected ['a', 'b'], got: {out.sub_queries}"
assert out.modality == "text", f"Expected 'text', got: {out.modality}"
print(f"  clean_query : {out.clean_query}")
print(f"  PASS\n")

print("[TEST 2] Reformer code fence stripping (```json ... ```)")
text = '```json\n{"clean_query": "question", "modality": "visual"}\n```'
out = reformer._parse(text)
assert out.clean_query == "question", f"Expected 'question', got: {out.clean_query}"
assert out.modality == "visual", f"Expected 'visual', got: {out.modality}"
print(f"  clean_query : {out.clean_query}")
print(f"  modality    : {out.modality}")
print(f"  PASS\n")

print("[TEST 3] Reformer malformed JSON fallback")
text = "this is not valid json"
out = reformer._parse(text)
assert isinstance(out, ReformerOutput), f"Did not return ReformerOutput: {type(out)}"
assert out.clean_query == "this is not valid json", f"Fallback incorrect: {out.clean_query}"
print(f"  clean_query : {out.clean_query}")
print(f"  PASS\n")


# VALIDATOR PARSE
print("[TEST 4] Validator PASS parse")
text = '{"verdict": true, "failure_type": "", "reason": ""}'
out = validator._parse(text)
assert out.verdict is True, f"verdict should be True, got: {out.verdict}"
print(f"  verdict : {out.verdict}")
print(f"  PASS\n")

print("[TEST 5] Validator FAIL parse")
text = '{"verdict": false, "failure_type": "grounding", "reason": "fabrication"}'
out = validator._parse(text)
assert out.verdict is False, f"verdict should be False, got: {out.verdict}"
assert out.failure_type == "grounding", f"failure_type should be 'grounding', got: {out.failure_type}"
print(f"  verdict      : {out.verdict}")
print(f"  failure_type : {out.failure_type}")
print(f"  PASS\n")

print("[TEST 6] Validator malformed JSON - safe fail")
text = "{broken"
out = validator._parse(text)
assert out.verdict is False, f"For malformed JSON verdict should be False, got: {out.verdict}"
print(f"  verdict : {out.verdict}")
print(f"  PASS\n")

print("All tests passed.")
