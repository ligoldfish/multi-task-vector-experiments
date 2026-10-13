import pytest
import torch

from task_vector.gao import gao_step


def _parameters():
    return [torch.nn.Parameter(torch.tensor([[.2, -.1], [.3, .4]])),
            torch.nn.Parameter(torch.tensor([.05, -.2]))]


def _groups():
    generator = torch.Generator().manual_seed(31)
    items = [{"id": str(i), "x": torch.randn(2, generator=generator),
              "y": torch.randn(2, generator=generator)} for i in range(16)]
    return items[:8], items[8:]


def _loss(params, group):
    x, y = torch.stack([row["x"] for row in group]), torch.stack([row["y"] for row in group])
    return ((x @ params[0].T + params[1] - y) ** 2).mean(dim=1).mean()


def _official_reference(params, optimizer, group1, group2, seed, rho_max=.05):
    rho = -.05 * float(torch.rand((), generator=torch.Generator().manual_seed(seed)))
    for source, target in ((group1, group2), (group2, group1)):
        gradients = torch.autograd.grad(_loss(params, source), params, create_graph=False)
        eps = [rho * grad / (grad.norm() + 1e-12) for grad in gradients]
        with torch.no_grad():
            for p, e in zip(params, eps):
                p.add_(e)
        optimizer.zero_grad(set_to_none=True)
        _loss(params, target).backward()
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for p, e in zip(params, eps):
                p.sub_(e)
    return rho


def test_exact_official_bidirectional_sequential_adamw_math():
    params, reference = _parameters(), _parameters()
    group1, group2 = _groups()
    optimizer = torch.optim.AdamW(params, lr=.001, weight_decay=.03)
    ref_optimizer = torch.optim.AdamW(reference, lr=.001, weight_decay=.03)
    expected_rho = _official_reference(reference, ref_optimizer, group1, group2, seed=17)
    result = gao_step(params, optimizer, group1, group2, lambda group: _loss(params, group),
                      generator=torch.Generator().manual_seed(17))
    assert result["rho"] == expected_rho
    assert result["optimizer_steps"] == 2
    assert -.05 <= result["rho"] <= 0
    for p, expected in zip(params, reference):
        assert torch.allclose(p, expected, atol=1e-8)
        assert optimizer.state[p]["step"] == 2
        assert p.grad is None


def test_micro2_group8_accumulation_matches_full_group_and_scheduler():
    full, micro = _parameters(), _parameters()
    group1, group2 = _groups()
    opt_full, opt_micro = torch.optim.AdamW(full, lr=.001, weight_decay=0), torch.optim.AdamW(micro, lr=.001, weight_decay=0)
    ticks = []

    def backward_micro(group):
        value = 0.
        for offset in range(0, len(group), 2):
            chunk = group[offset:offset + 2]
            loss = _loss(micro, chunk) * len(chunk) / len(group)
            loss.backward()
            value += float(loss.detach())
        return value

    full_result = gao_step(full, opt_full, group1, group2, lambda group: _loss(full, group),
                           generator=torch.Generator().manual_seed(29))
    micro_result = gao_step(micro, opt_micro, group1, group2, backward_micro,
                            generator=torch.Generator().manual_seed(29), scheduler=lambda: ticks.append(1))
    for p, expected in zip(micro, full):
        assert torch.allclose(p, expected, atol=1e-7)
    assert full_result["loss"] == pytest.approx(micro_result["loss"], rel=1e-6)
    assert len(ticks) == micro_result["scheduler_steps"] == 2


def test_target_failure_always_removes_perturbation():
    params = _parameters()
    originals = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=.001)
    group1, group2 = _groups()
    calls = []

    def fail_target(group):
        calls.append(group)
        if len(calls) == 2:
            raise RuntimeError("simulated forward failure")
        return _loss(params, group)

    with pytest.raises(RuntimeError, match="simulated"):
        gao_step(params, optimizer, group1, group2, fail_target)
    for p, original in zip(params, originals):
        assert torch.allclose(p, original, atol=3e-8)
    assert not optimizer.state


def test_groups_with_duplicate_or_overlap_ids_fail_before_update():
    params = _parameters()
    group1, group2 = _groups()
    group2[0]["id"] = group1[0]["id"]
    with pytest.raises(ValueError, match="disjoint"):
        gao_step(params, torch.optim.AdamW(params), group1, group2, lambda group: _loss(params, group))
