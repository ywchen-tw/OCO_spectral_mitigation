from __future__ import annotations

import sys
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path

import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ``models.__init__`` eagerly imports TabM, whose progress-bar dependency is not
# needed by these architecture-only tests and is optional in minimal CI images.
try:
    import tqdm  # noqa: F401
except ImportError:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.__spec__ = ModuleSpec("tqdm", loader=None)
    tqdm_module.tqdm = lambda iterable=None, *args, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_module

from models.dcn_v2 import DCNV2Regressor, LowRankCrossLayer  # noqa: E402
from models.structured_dcn_ensemble import build_experimental_member  # noqa: E402
from models.structured_residual import (  # noqa: E402
    StructuredResidualNet,
    group_feature_indices,
)


FEATURE_NAMES = [
    "xco2_raw_minus_apriori",
    "o2a_k1_nosg",
    "aod_dust",
    "log_P",
    "t_pc01",
    "tropopause_sigma",
    "1_over_cos_sza",
    "fp_0",
    "fp_1",
]


class StructuredResidualTests(unittest.TestCase):
    def test_feature_groups_are_exhaustive_and_disjoint(self) -> None:
        groups = group_feature_indices(FEATURE_NAMES)
        flattened = [idx for indices in groups.values() for idx in indices]
        self.assertEqual(sorted(flattened), list(range(len(FEATURE_NAMES))))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(groups["xco2"], [0])
        self.assertEqual(groups["spectroscopy"], [1])
        self.assertEqual(groups["profile"], [4, 5])

    def test_no_spec_feature_set_omits_spectroscopy_encoder(self) -> None:
        names = [name for name in FEATURE_NAMES if name != "o2a_k1_nosg"]
        model = StructuredResidualNet(
            names,
            hidden_dims=(12, 6),
            block_dim=4,
            norm="layer",
        )
        self.assertNotIn("spectroscopy", model.feature_groups)
        mu, raw2 = model(torch.randn(7, len(names)))
        self.assertEqual(mu.shape, (7,))
        self.assertEqual(raw2.shape, (7,))

    def test_auxiliary_head_and_gradient(self) -> None:
        model = StructuredResidualNet(
            FEATURE_NAMES,
            hidden_dims=(12, 6),
            block_dim=4,
            aux_cloud=True,
            dropout=0.1,
            norm="layer",
        )
        outputs = model(torch.randn(8, len(FEATURE_NAMES)))
        self.assertEqual([tuple(out.shape) for out in outputs], [(8,), (8,), (8,)])
        sum(out.mean() for out in outputs).backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.parameters())
        )


class DCNV2Tests(unittest.TestCase):
    def test_zero_cross_transform_is_identity(self) -> None:
        layer = LowRankCrossLayer(n_features=5, rank=3)
        for parameter in layer.parameters():
            torch.nn.init.zeros_(parameter)
        x0 = torch.randn(4, 5)
        x = torch.randn(4, 5)
        self.assertTrue(torch.equal(layer(x0, x), x))

    def test_output_shapes_and_gradient(self) -> None:
        model = DCNV2Regressor(
            n_features=9,
            hidden_dims=(12, 6),
            cross_layers=2,
            cross_rank=4,
            aux_cloud=True,
            norm="layer",
        )
        outputs = model(torch.randn(8, 9))
        self.assertEqual([tuple(out.shape) for out in outputs], [(8,), (8,), (8,)])
        sum(out.mean() for out in outputs).backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.parameters())
        )

    def test_shared_factory_builds_each_backbone(self) -> None:
        for backbone in ("structured_residual", "dcn_v2"):
            model = build_experimental_member(
                backbone,
                len(FEATURE_NAMES),
                feature_names=FEATURE_NAMES,
                hidden_dims=(12, 6),
                block_dim=4,
                cross_layers=2,
                cross_rank=4,
            )
            x = torch.randn(3, len(FEATURE_NAMES))
            mu, raw2 = model(x)
            self.assertEqual(mu.shape, (3,))
            self.assertEqual(raw2.shape, (3,))

            restored = build_experimental_member(
                backbone,
                len(FEATURE_NAMES),
                feature_names=FEATURE_NAMES,
                hidden_dims=(12, 6),
                block_dim=4,
                cross_layers=2,
                cross_rank=4,
            )
            restored.load_state_dict(model.state_dict())
            restored_mu, restored_raw2 = restored(x)
            self.assertTrue(torch.equal(mu, restored_mu))
            self.assertTrue(torch.equal(raw2, restored_raw2))


if __name__ == "__main__":
    unittest.main()
