"""CPU-only tests for the PE (Prompt Enhancement) prompt-rewrite path.

Two layers:

* The torch-free text helpers (:func:`extract_pe_text` /
  :func:`postprocess_pe_texts`) shared by the sglang ``ComposedRolloutEngine``
  and the trainside :class:`PEPipeline` — marker extraction, ``<think>`` strip,
  quote strip, truncation, and original-prompt fallback.
* :class:`PEPipeline` injecting ``pe_instruction`` into the LLM child's chat
  ``system_instruction`` and stripping ``pe_marker`` from the LLM output before
  the diffusion child conditions on it — mirroring the sglang engine
  (``composed/engine.py``).

The pipeline-level tests drive ``PEPipeline.generate`` with fake child pipelines
(no model weights, no GPU), asserting the request the LLM child receives and the
text the diffusion child receives.
"""

from types import SimpleNamespace

from unirl.models.pe.instruction import extract_pe_text, postprocess_pe_texts
from unirl.models.pe.pipeline import PEPipeline
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack

MARKER = "Revised Prompt:"


# --------------------------------------------------------------------------
# Text helpers (torch-free)
# --------------------------------------------------------------------------


def test_extract_pe_text_after_marker_with_quotes():
    raw = 'My reasoning about the scene. Revised Prompt:\n"a red cat on a sofa"'
    assert extract_pe_text(raw, MARKER) == "a red cat on a sofa"


def test_extract_pe_text_strips_think_preamble():
    raw = "<think>step by step</think> Revised Prompt: a blue dog"
    assert extract_pe_text(raw, MARKER) == "a blue dog"


def test_extract_pe_text_uses_last_marker_occurrence():
    raw = "Revised Prompt: draft one\nActually, Revised Prompt: final version"
    assert extract_pe_text(raw, MARKER) == "final version"


def test_extract_pe_text_missing_marker_returns_empty():
    assert extract_pe_text("no marker here at all", MARKER) == ""


def test_extract_pe_text_empty_input_returns_empty():
    assert extract_pe_text("", MARKER) == ""
    assert extract_pe_text("   ", MARKER) == ""


def test_postprocess_falls_back_to_original_prompt_on_off_format():
    raw = ["just rambled, no marker", "x Revised Prompt: a clean rewrite"]
    cleaned, stats = postprocess_pe_texts(
        raw,
        user_prompts=["orig prompt 0", "orig prompt 1"],
        samples_per_prompt=1,
        marker=MARKER,
    )
    assert cleaned == ["orig prompt 0", "a clean rewrite"]
    assert stats["empty"] == 1
    assert stats["fallback"] == 1


def test_postprocess_maps_pe_major_slots_to_user_prompts():
    # 2 prompts x N=2 rewrites, PE-major: slots 0,1 -> prompt0; slots 2,3 -> prompt1.
    raw = ["bad", "bad", "Revised Prompt: ok", "bad"]
    cleaned, stats = postprocess_pe_texts(
        raw,
        user_prompts=["P0", "P1"],
        samples_per_prompt=2,
        marker=MARKER,
    )
    assert cleaned == ["P0", "P0", "ok", "P1"]
    assert stats["fallback"] == 3


def test_postprocess_truncates_to_max_chars():
    cleaned, stats = postprocess_pe_texts(
        ["x Revised Prompt: abcdefghij"],
        user_prompts=["orig"],
        samples_per_prompt=1,
        marker=MARKER,
        max_chars=5,
    )
    assert cleaned == ["abcde"]
    assert stats["truncated"] == 1


# --------------------------------------------------------------------------
# PEPipeline injection + stripping
# --------------------------------------------------------------------------


class _FakeLLMPipeline:
    """Records the request it receives; echoes back canned rewrite text."""

    def __init__(self, rewrites):
        self.bundle = SimpleNamespace()
        self._rewrites = rewrites
        self.last_req = None

    def generate(self, req):
        self.last_req = req
        n = len(req.sample_ids)
        track = RolloutTrack(
            sample_ids=list(req.sample_ids),
            parent_ids=list(req.group_ids),
            parent_track=None,
            conditions={},
            segment=None,
            decoded=Texts(texts=list(self._rewrites[:n])),
        )
        return RolloutResp(tracks={"ar": track})


class _FakeDiffusionPipeline:
    """Records the prompts it receives; emits a stub image track."""

    def __init__(self):
        self.bundle = SimpleNamespace()
        self.last_texts = None

    def generate(self, req):
        self.last_texts = list(req.primitives["text"].texts)
        track = RolloutTrack(
            sample_ids=list(req.sample_ids),
            parent_ids=list(req.group_ids),
            parent_track=None,
            conditions={},
            segment=None,
            decoded=Images(pixels=None),
        )
        return RolloutResp(tracks={"image": track})


def _make_req(prompts, *, n_rewrites, n_images):
    sampling = SimpleNamespace(
        ar=SimpleNamespace(samples_per_prompt=n_rewrites),
        diffusion=SimpleNamespace(samples_per_prompt=n_images),
    )
    return RolloutReq(
        sample_ids=[f"p{i}" for i in range(len(prompts))],
        group_ids=[f"p{i}" for i in range(len(prompts))],
        primitives={"text": Texts(texts=list(prompts))},
        request_conditions={},
        sampling_params=sampling,
        stage_config={},
    )


def test_pipeline_injects_pe_instruction_into_llm_chat():
    llm = _FakeLLMPipeline(rewrites=["x Revised Prompt: r0", "x Revised Prompt: r1"])
    diff = _FakeDiffusionPipeline()
    pipe = PEPipeline(
        diffusion_pipeline=diff,
        llm_pipeline=llm,
        pe_instruction="You are a Prompt Optimizer.",
        pe_marker=MARKER,
    )
    req = _make_req(["a cat"], n_rewrites=2, n_images=1)
    pipe.generate(req)

    assert llm.last_req.stage_config["chat"]["system_instruction"] == "You are a Prompt Optimizer."


def test_pipeline_strips_marker_before_diffusion():
    # Rewrites carry reasoning preambles; SD3 must see only the cleaned text.
    llm = _FakeLLMPipeline(rewrites=["blah blah Revised Prompt: a crisp red cat"])
    diff = _FakeDiffusionPipeline()
    pipe = PEPipeline(
        diffusion_pipeline=diff,
        llm_pipeline=llm,
        pe_instruction="instr",
        pe_marker=MARKER,
    )
    req = _make_req(["a cat"], n_rewrites=1, n_images=2)
    resp = pipe.generate(req)

    # Diffusion conditions on the cleaned rewrite, M-replicated.
    assert diff.last_texts == ["a crisp red cat", "a crisp red cat"]
    # The AR track's decoded is rewritten in place so logging sees clean text.
    assert resp.tracks["ar"].decoded.texts == ["a crisp red cat"]


def test_pipeline_marker_fallback_to_original_prompt():
    llm = _FakeLLMPipeline(rewrites=["the model forgot the marker entirely"])
    diff = _FakeDiffusionPipeline()
    pipe = PEPipeline(
        diffusion_pipeline=diff,
        llm_pipeline=llm,
        pe_instruction="instr",
        pe_marker=MARKER,
    )
    req = _make_req(["original cat prompt"], n_rewrites=1, n_images=1)
    pipe.generate(req)

    assert diff.last_texts == ["original cat prompt"]


def test_pipeline_without_pe_knobs_forwards_text_verbatim():
    # No instruction, no marker → prior behavior: raw LLM text reaches diffusion,
    # and the LLM child gets no injected system_instruction.
    llm = _FakeLLMPipeline(rewrites=["raw llm continuation"])
    diff = _FakeDiffusionPipeline()
    pipe = PEPipeline(diffusion_pipeline=diff, llm_pipeline=llm)
    req = _make_req(["a cat"], n_rewrites=1, n_images=1)
    pipe.generate(req)

    assert diff.last_texts == ["raw llm continuation"]
    assert "chat" not in llm.last_req.stage_config
