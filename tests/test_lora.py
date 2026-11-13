import numpy as np
import torch

from encoder.export import build_backbone
from encoder.lora import LoRAConv2d, add_lora, sample_fraction


def test_lora_starts_as_the_frozen_model():
    torch.manual_seed(0)
    plain = build_backbone("random").eval()
    adapted = build_backbone("random").eval()
    adapted.load_state_dict(plain.state_dict())
    add_lora(adapted, r=8)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(plain(x), adapted(x), atol=1e-6)  # B starts at zero


def test_only_lora_weights_train():
    bb = build_backbone("random")
    params = add_lora(bb, r=8)
    trainable = [n for n, p in bb.named_parameters() if p.requires_grad]
    assert len(params) == 20 and all(".A." in n or ".B." in n for n in trainable)  # 10 convs in layer4, A and B each
    assert sum(p.numel() for p in params) < 0.02 * sum(p.numel() for p in bb.parameters())


def test_lora_branch_keeps_shape_and_stride():
    conv = torch.nn.Conv2d(512, 512, 3, stride=2, padding=1)
    m = LoRAConv2d(conv, r=4)
    torch.nn.init.normal_(m.B.weight)
    assert m(torch.randn(1, 512, 14, 14)).shape == (1, 512, 7, 7)


def test_sample_fraction_is_stratified_and_seeded():
    rng = np.random.default_rng(0)
    Y = (rng.random((1000, 5)) < [0.5, 0.2, 0.1, 0.05, 0.02]).astype(int)
    a, b = sample_fraction(Y, 0.1, 0), sample_fraction(Y, 0.1, 0)
    assert np.array_equal(a, b) and abs(len(a) - 100) <= 2
    assert np.allclose(Y[a].mean(0), Y.mean(0), atol=0.03)
    assert len(sample_fraction(Y, 1.0, 0)) == 1000
