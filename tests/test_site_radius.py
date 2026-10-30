from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "workspace"))
from tccon_collocate import (PRODUCTION_SITE_RADII, SITE_COORDS,
                             SITE_RADIUS_SCHEMES, collocate, site_radius_km)


class SiteRadiusTest(unittest.TestCase):
    def test_uniform_keeps_global_radius(self):
        for site in ("ci", "df", "jf", "tk", "oc"):
            self.assertEqual(site_radius_km(site, 100.0, "uniform"), 100.0)

    def test_das2025_overrides(self):
        self.assertEqual(site_radius_km("ci", 100.0, "das2025"), 25.0)
        self.assertEqual(site_radius_km("jf", 100.0, "das2025"), 25.0)
        self.assertEqual(site_radius_km("df", 100.0, "das2025"), 50.0)
        self.assertEqual(site_radius_km("tk", 100.0, "das2025"), 100.0)
        self.assertEqual(site_radius_km("tk", 100.0, "das2025_tk50"), 50.0)
        self.assertEqual(site_radius_km("tk", 100.0, "das2025_tk25"), 25.0)
        for site in ("ci", "jf", "df", "tk"):
            self.assertEqual(site_radius_km(site, 100.0, "site25"), 25.0)

    def test_override_only_tightens(self):
        self.assertEqual(site_radius_km("df", 30.0, "das2025"), 30.0)
        self.assertEqual(site_radius_km("ci", 30.0, "das2025"), 25.0)

    def test_accepts_tccon_file_prefix(self):
        self.assertEqual(site_radius_km("CI20120920", 100.0, "das2025"), 25.0)

    def test_every_override_site_has_coordinates(self):
        for scheme in SITE_RADIUS_SCHEMES.values():
            for site in scheme:
                self.assertIn(site, SITE_COORDS)

    def test_collocate_respects_radius(self):
        st_lon, st_lat = SITE_COORDS["ci"]
        # Footprints due north of Caltech at 10, 40 and 80 km.
        lat = st_lat + np.array([10.0, 40.0, 80.0]) / 111.195
        t0 = pd.Timestamp("2022-06-16 21:00", tz="UTC")
        oco = pd.DataFrame(dict(time=[t0.timestamp()] * 3, lon=[st_lon] * 3,
                                lat=lat, xco2_bc=[415.0] * 3))
        tccon = pd.DataFrame(dict(time=[t0], lon=[st_lon], lat=[st_lat],
                                  xco2=[415.0]))
        box = (st_lon - 1, st_lon + 1, st_lat - 1, st_lat + 1)

        def n_near(**kw):
            col = collocate(oco, tccon, box=box, radius_km=100.0, window_min=60,
                            site="ci", **kw)
            return len(col["near"]), col["radius_km"]

        self.assertEqual(n_near(site_radii="uniform"), (3, 100.0))
        self.assertEqual(n_near(site_radii="das2025"), (1, 25.0))
        # collocate() applies the production scheme unless told otherwise.
        self.assertEqual(n_near(), n_near(site_radii=PRODUCTION_SITE_RADII))
        self.assertEqual(n_near(), (1, 25.0))

    def test_production_scheme(self):
        self.assertEqual(PRODUCTION_SITE_RADII, "das2025_tk25")
        self.assertEqual(SITE_RADIUS_SCHEMES[PRODUCTION_SITE_RADII],
                         {"ci": 25.0, "jf": 25.0, "df": 50.0, "tk": 25.0})


if __name__ == "__main__":
    unittest.main()
