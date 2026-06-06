from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.phase_01_metadata import OCO2MetadataRetriever  # noqa: E402
from pipeline.phase_02_ingestion import DataIngestionManager, DownloadedFile  # noqa: E402


class FakeResponse:
    def __init__(
        self,
        text: str = "",
        status_code: int = 200,
        url: str = "https://example.test/",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.text = text
        self.status_code = status_code
        self.url = url
        self.headers = headers or {"Content-Type": "text/html"}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self) -> None:
        pass


class Phase2IngestionResilienceTests(unittest.TestCase):
    def test_cmr_fallback_starts_with_short_name_and_version(self) -> None:
        feed = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry><title>oco2_L1bScGL_12345a_160915_B11006r_000000.h5</title></entry>
        </feed>
        """
        calls: list[dict[str, object]] = []

        def fake_get(url: str, params: dict[str, object], timeout: int) -> FakeResponse:
            calls.append(params)
            return FakeResponse(text=feed, status_code=200, url=url)

        env = {"EARTHDATA_USERNAME": "", "EARTHDATA_PASSWORD": "", "LAADS_TOKEN": ""}
        with mock.patch.dict(os.environ, env):
            retriever = OCO2MetadataRetriever()
        with mock.patch("pipeline.phase_01_metadata.requests.get", side_effect=fake_get):
            xml = retriever.fetch_oco2_xml_from_cmr(
                __import__("datetime").datetime(2016, 9, 15)
            )

        self.assertEqual(xml, feed)
        self.assertEqual(calls[0]["short_name"], "OCO2_L1B_Science")
        self.assertEqual(calls[0]["version"], "11r")
        self.assertEqual(calls[0]["provider"], "GES_DISC")

    def test_gesdisc_directory_falls_back_from_token_to_plain_data_url(self) -> None:
        env = {"EARTHDATA_USERNAME": "", "EARTHDATA_PASSWORD": "", "LAADS_TOKEN": ""}
        with mock.patch.dict(os.environ, env):
            retriever = OCO2MetadataRetriever()
        calls: list[str] = []

        def fake_get_with_retry(url: str) -> FakeResponse:
            calls.append(url)
            if "/data/.expired-token/" in url:
                return FakeResponse(
                    text="<html>Earthdata Login</html>",
                    status_code=200,
                    url="https://urs.earthdata.nasa.gov/oauth/authorize",
                )
            if url.endswith("/2016/259/"):
                return FakeResponse(
                    text='<html><a href="sample.xml">sample.xml</a></html>',
                    status_code=200,
                    url=url,
                )
            return FakeResponse(
                text="<S4PAGranuleMetaDataFile></S4PAGranuleMetaDataFile>",
                status_code=200,
                url=url,
                headers={"Content-Type": "application/xml"},
            )

        with mock.patch.dict(os.environ, {"GESDISC_DATA_TOKEN": ".expired-token"}):
            with mock.patch.object(retriever, "_get_with_retry", side_effect=fake_get_with_retry):
                xml_files = retriever.fetch_oco2_xml_from_directory(
                    __import__("datetime").datetime(2016, 9, 15)
                )

        self.assertEqual(len(xml_files), 1)
        self.assertTrue(any("/data/.expired-token/" in url for url in calls))
        self.assertTrue(any("/data/OCO2_DATA/" in url for url in calls))

    def test_gesdisc_directory_with_oauth_text_is_not_login_page(self) -> None:
        response = FakeResponse(
            text=(
                '<html><a href="oco2_L1bScGL_11734a_160915_B11006r_221129020008.h5.xml">'
                "metadata</a><!-- references urs.earthdata.nasa.gov/oauth docs --></html>"
            ),
            status_code=200,
            url="https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L1B_Science.11r/2016/259/",
        )

        self.assertFalse(OCO2MetadataRetriever._looks_like_auth_page(response))
        self.assertFalse(DataIngestionManager._looks_like_auth_page(response))

    def test_cmr_atom_entry_parses_time_namespace(self) -> None:
        entry_xml = """<entry
            xmlns="http://www.w3.org/2005/Atom"
            xmlns:echo="http://www.echo.nasa.gov/esip"
            xmlns:time="http://a9.com/-/opensearch/extensions/time/1.0/">
          <title>oco2_L1bScGL_11734a_160915_B11006r_221129020008.h5</title>
          <echo:producerGranuleId>oco2_L1bScGL_11734a_160915_B11006r_221129020008.h5</echo:producerGranuleId>
          <time:start>2016-09-15T00:00:00.000Z</time:start>
          <time:end>2016-09-15T00:06:00.000Z</time:end>
          <link rel="http://esipfed.org/ns/fedsearch/1.1/data#"
                href="https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L1B_Science.11r/2016/259/oco2_L1bScGL_11734a_160915_B11006r_221129020008.h5"/>
        </entry>"""
        namespaces = {
            "atom": "http://www.w3.org/2005/Atom",
            "echo": "http://www.echo.nasa.gov/esip",
            "time": "http://a9.com/-/opensearch/extensions/time/1.0/",
        }
        env = {"EARTHDATA_USERNAME": "", "EARTHDATA_PASSWORD": "", "LAADS_TOKEN": ""}
        with mock.patch.dict(os.environ, env):
            retriever = OCO2MetadataRetriever()

        granule = retriever._parse_atom_entry(ET.fromstring(entry_xml), namespaces)

        self.assertIsNotNone(granule)
        self.assertEqual(granule.orbit_str, "11734a")
        self.assertEqual(granule.viewing_mode, "GL")
        self.assertEqual(granule.start_time.year, 2016)
        self.assertTrue(granule.download_url.endswith(".h5"))

    def test_status_file_requires_l1b_and_no_failed_downloads(self) -> None:
        granule_id = "oco2_L1bScGL_12345a_160915_B11006r_000000.h5"
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {"EARTHDATA_USERNAME": "", "EARTHDATA_PASSWORD": "", "LAADS_TOKEN": ""}
            with mock.patch.dict(os.environ, env):
                manager = DataIngestionManager(output_dir=tmpdir)
            l1b_path = (
                Path(tmpdir)
                / "OCO2"
                / "2016"
                / "259"
                / "12345a_GL"
                / granule_id
            )
            downloaded = DownloadedFile(
                filepath=l1b_path,
                product_type="OCO2_L1B",
                target_year=2016,
                target_doy=259,
                granule_id=granule_id,
                file_size_mb=1.0,
                download_time_seconds=0.0,
            )
            target_date = __import__("datetime").datetime(2016, 9, 15)

            manager._write_download_status(target_date, [granule_id], [downloaded], [])
            status_path = l1b_path.parent / "sat_data_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertTrue(status["downloading_completed"])

            manager.download_stats["failed_downloads"].append(
                {"granule_id": granule_id, "product_type": "L2_Lite", "url": "N/A"}
            )
            manager._write_download_status(target_date, [granule_id], [downloaded], [])
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertFalse(status["downloading_completed"])
            self.assertEqual(status["failed_download_count"], 1)

    def test_cmr_product_link_matching_for_lite_met_and_prior(self) -> None:
        atom = """<feed xmlns="http://www.w3.org/2005/Atom"
                       xmlns:echo="http://www.echo.nasa.gov/esip">
          <entry>
            <title>oco2_LtCO2_180101_B11100Ar_230602044234s.nc4</title>
            <link rel="http://esipfed.org/ns/fedsearch/1.1/data#"
                  href="https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L2_Lite_FP.11.1r/2018/oco2_LtCO2_180101_B11100Ar_230602044234s.nc4"/>
          </entry>
          <entry>
            <title>oco2_L2MetGL_18636a_180101_B11006r_221009082010.h5</title>
            <link rel="http://esipfed.org/ns/fedsearch/1.1/data#"
                  href="https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L2_Met.11r/2018/001/oco2_L2MetGL_18636a_180101_B11006r_221009082010.h5"/>
          </entry>
          <entry>
            <title>oco2_L2CPrGL_18636a_180101_B11006r_221009082010.h5</title>
            <link rel="http://esipfed.org/ns/fedsearch/1.1/data#"
                  href="https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L2_CO2Prior.11r/2018/001/oco2_L2CPrGL_18636a_180101_B11006r_221009082010.h5"/>
          </entry>
        </feed>"""
        granule_id = "oco2_L1bScGL_18636a_180101_B11006r_221009082010.h5"
        env = {"EARTHDATA_USERNAME": "", "EARTHDATA_PASSWORD": "", "LAADS_TOKEN": ""}
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, env):
                manager = DataIngestionManager(output_dir=tmpdir)

            lite_urls = manager._extract_matching_product_urls_from_cmr(
                atom,
                product_type="L2_Lite",
                granule_id=granule_id,
                target_date_str="180101",
            )
            met_urls = manager._extract_matching_product_urls_from_cmr(
                atom,
                product_type="L2_Met",
                granule_id=granule_id,
                target_date_str="180101",
            )
            prior_urls = manager._extract_matching_product_urls_from_cmr(
                atom,
                product_type="L2_CO2Prior",
                granule_id=granule_id,
                target_date_str="180101",
            )

        self.assertEqual(len(lite_urls), 1)
        self.assertIn("LtCO2_180101", lite_urls[0])
        self.assertEqual(len(met_urls), 1)
        self.assertIn("L2MetGL_18636a", met_urls[0])
        self.assertEqual(len(prior_urls), 1)
        self.assertIn("L2CPrGL_18636a", prior_urls[0])


if __name__ == "__main__":
    unittest.main()
