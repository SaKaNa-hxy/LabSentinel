import unittest

from scripts.crop_upperbody_dataset import plan_crop, transform_boxes


class UpperBodyCropTests(unittest.TestCase):
    def test_all_classes_transform_and_outside_targets_are_removed(self):
        boxes = [[5, 20, 20, 100, 180], [6, 30, 10, 50, 30], [2, 40, 40, 60, 60], [7, 130, 130, 150, 150]]
        labels, reason = transform_boxes(boxes, [10, 0, 120, 120], 0)
        self.assertIsNone(reason)
        self.assertEqual([row[0] for row in labels], [5, 6, 2])
        self.assertAlmostEqual(labels[0][1], 50/110)
        self.assertAlmostEqual(labels[0][2], 70/120)
        self.assertAlmostEqual(labels[0][3], 80/110)
        self.assertAlmostEqual(labels[0][4], 100/120)

    def test_partly_cut_action_rejects_crop_instead_of_leaving_it_unlabelled(self):
        labels, reason = transform_boxes([[5, 0, 0, 100, 200], [0, 40, 80, 80, 140]], [0, 0, 100, 120], 0)
        self.assertIsNone(labels)
        self.assertEqual(reason, "ambiguous_partial_object")

    def test_partial_small_ppe_rejects_crop(self):
        labels, reason = transform_boxes([[9, 0, 0, 100, 200], [2, 40, 100, 80, 150]], [0, 0, 100, 120], 0)
        self.assertIsNone(labels)
        self.assertEqual(reason, "ambiguous_partial_object")

    def test_planner_preserves_head_context_and_excludes_lower_coat(self):
        plan, reason = plan_crop([[5, 200, 180, 420, 600], [8, 240, 30, 360, 150]], 0, 640, 640)
        self.assertIsNone(reason)
        self.assertEqual(plan["crop_xyxy"][1], 0)
        self.assertLess(plan["crop_xyxy"][3], 600)
        self.assertEqual({row[0] for row in plan["labels"]}, {5, 8})

    def test_tiny_garments_are_skipped(self):
        plan, reason = plan_crop([[5, 0, 0, 20, 40]], 0, 640, 640)
        self.assertIsNone(plan)
        self.assertEqual(reason, "small_garment")


if __name__ == "__main__":
    unittest.main()
