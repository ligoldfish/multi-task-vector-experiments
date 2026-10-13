import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from task_vector.evaluation import greedy_generate
from task_vector.losses import answer_nll
from task_vector.ltv import LTVAdapter, LTVIntervention, intervention_context
from task_vector.representations import LowRankFactor, Representation


def tiny_model():
    torch.manual_seed(31)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=37, hidden_size=16,
        intermediate_size=24, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, head_dim=4, attention_dropout=0.0,
        max_position_embeddings=256, eos_token_id=2, pad_token_id=0))
    model.eval()
    model.requires_grad_(False)
    return model


def test_hook_changes_only_original_prompt_position_before_block():
    model = tiny_model()
    ids = torch.tensor([[4, 5, 6, 7, 8], [3, 5, 7, 0, 0]])
    inputs = []
    handle = model.model.layers[1].register_forward_pre_hook(lambda m, args: inputs.append(args[0].detach().clone()))
    model(ids, use_cache=False)
    handle.remove()
    original = inputs.pop()
    rep = Representation("activation", "test", 17, values={"model.layers.1": torch.arange(16).float() / 100})
    with LTVIntervention(model, rep) as ctx:
        ctx.set_positions(torch.tensor([2, 1]))
        handle = model.model.layers[1].register_forward_pre_hook(lambda m, args: inputs.append(args[0].detach().clone()))
        model(ids, use_cache=False)
        handle.remove()
    expected = torch.zeros_like(original)
    expected[0, 2] = rep.values["model.layers.1"]
    expected[1, 1] = rep.values["model.layers.1"]
    torch.testing.assert_close(inputs[0] - original, expected, atol=1e-8, rtol=1e-5)


def test_cached_decode_equals_recomputed_fixed_prompt_injection():
    model = tiny_model()
    ids = torch.tensor([[4, 5, 6], [8, 9, 10]])
    adapter = LTVAdapter(model, [1], seed=17)
    with adapter:
        adapter.set_positions(torch.tensor([2, 2]))
        prefill = model(ids, use_cache=True)
        token = prefill.logits[:, -1].argmax(-1)[:, None]
        # Automatic detection: do not set prefill=False explicitly.
        step = model(token, past_key_values=prefill.past_key_values,
                     attention_mask=torch.ones((2, 4), dtype=torch.long), use_cache=True)
        whole = model(torch.cat([ids, token], dim=1), use_cache=False)
    torch.testing.assert_close(step.logits[:, -1], whole.logits[:, -1], atol=2e-6, rtol=2e-5)


def test_nonreentrant_checkpoint_recomputes_identical_injection_and_gradient():
    model = tiny_model()
    adapter = LTVAdapter(model, [1], seed=17)
    ids = torch.tensor([[4, 5, 6, 7, 8], [3, 5, 7, 9, 2]])
    labels = torch.tensor([[-100, -100, -100, 7, 8], [-100, -100, 7, 9, 2]])
    with adapter:
        adapter.set_positions(torch.tensor([2, 1]))
        model.train()
        base_loss = answer_nll(model(ids, use_cache=False).logits, labels).mean()
        base_loss.backward()
        wanted = adapter.vectors["1"].grad.clone()
        adapter.zero_grad(set_to_none=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        loss = answer_nll(model(ids, use_cache=False).logits, labels).mean()
        loss.backward()
        torch.testing.assert_close(loss, base_loss, atol=0, rtol=0)
        torch.testing.assert_close(adapter.vectors["1"].grad, wanted, atol=1e-7, rtol=1e-5)
        assert wanted.norm() > 0


def test_multilayer_independent_parameters_and_exact_save_reload(tmp_path):
    model = tiny_model()
    adapter = LTVAdapter(model, [0, 1, 2], seed=17)
    assert sum(p.numel() for p in adapter.parameters()) == 3 * 16
    assert not torch.equal(adapter.vectors["0"], adapter.vectors["1"])
    rep = adapter.representation("math", 17)
    rep.save(tmp_path / "ltv.pt")
    reloaded = Representation.load(tmp_path / "ltv.pt")
    ids = torch.tensor([[3, 4, 5]])
    with intervention_context(model, rep, torch.tensor([2])):
        before = model(ids, use_cache=False).logits
    with intervention_context(model, reloaded, torch.tensor([2])):
        after = model(ids, use_cache=False).logits
    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_parameter_wrapper_equals_dense_ba_and_restores_native_projection():
    model = tiny_model()
    reference = copy.deepcopy(model)
    key = "model.layers.1.self_attn.k_proj"
    original = model.get_submodule(key)
    torch.manual_seed(21)
    B = torch.randn(original.out_features, 3) * 0.01
    A = torch.randn(3, original.in_features) * 0.01
    rep = Representation("parameter", "math", 17, factors={key: [LowRankFactor(B, A)]})
    with torch.no_grad():
        reference.get_submodule(key).weight.add_(B @ A)
    ids = torch.tensor([[3, 4, 5, 6]])
    with intervention_context(model, rep):
        actual = model(ids, use_cache=False).logits
    expected = reference(ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    assert model.get_submodule(key) is original


def test_left_padded_greedy_equals_individual_and_records_length_stop():
    model = tiny_model()
    adapter = LTVAdapter(model, [1], seed=17)
    rep = adapter.representation("math", 17)
    ids = torch.tensor([[0, 0, 4, 5], [3, 6, 7, 8]])
    mask = ids.ne(0).long()
    with intervention_context(model, rep) as ctx:
        together, ended = greedy_generate(model, ids, mask, max_new_tokens=3,
            intervention=ctx, eos_token_ids=[99], pad_token_id=0)
    assert ended == [False, False]
    for i, prefix in enumerate(([4, 5], [3, 6, 7, 8])):
        one = torch.tensor([prefix])
        with intervention_context(model, rep) as ctx:
            alone, _ = greedy_generate(model, one, torch.ones_like(one), max_new_tokens=3,
                intervention=ctx, eos_token_ids=[99], pad_token_id=0)
        assert together[i] == alone[0]


def test_loading_rejects_wrong_gqa_even_with_correct_layer_and_hidden_sizes(monkeypatch):
    import transformers
    from task_vector.modeling import load_model
    fake = torch.nn.Linear(1, 1)
    fake.config = SimpleNamespace(model_type="qwen3", num_hidden_layers=28, hidden_size=2048,
                                  num_attention_heads=16, num_key_value_heads=16, head_dim=128)
    tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: fake)
    with pytest.raises(ValueError, match="num_key_value_heads"):
        load_model({"model": {"path": "local-test-fixture", "enable_thinking": False}}, "cpu")
