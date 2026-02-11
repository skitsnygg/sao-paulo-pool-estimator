import unittest

from tools.dedupe_geojson_polygons import dedupe_feature_collection


class TestDedupeGeojsonPolygons(unittest.TestCase):
    def test_suppression_with_overlap(self) -> None:
        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"score": 0.9, "id": "a"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
                        ],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {"score": 0.1, "id": "b"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[0.2, 0.2], [1.2, 0.2], [1.2, 1.2], [0.2, 1.2], [0.2, 0.2]]
                        ],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {"score": 0.5, "id": "c"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[3.0, 3.0], [4.0, 3.0], [4.0, 4.0], [3.0, 4.0], [3.0, 3.0]]
                        ],
                    },
                },
            ],
        }

        out_fc, stats = dedupe_feature_collection(
            fc,
            iou=0.3,
            score_key="score",
            precision=6,
        )

        self.assertEqual(stats.features_in, 3)
        self.assertEqual(stats.suppressed, 1)
        self.assertEqual(len(out_fc["features"]), 2)
        self.assertTrue(stats.top_overlap_ious)


if __name__ == "__main__":
    unittest.main()
