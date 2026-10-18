import contextlib
import io
import json
import unittest

import animate
import run_mission


class CommandLineTests(unittest.TestCase):
    def test_dynamic_json_report(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = run_mission.main(['--mode', 'dynamic', '--scenario', 'demo', '--json'])
        self.assertEqual(status, 0)
        report = json.loads(output.getvalue())
        self.assertTrue(report['arrived'])
        self.assertFalse(report['collision'])
        self.assertGreater(report['max_tilt_degrees'], 12)
        self.assertGreater(report['min_point_clearance_m'], .18)

    def test_invalid_mission_duration_is_usage_error(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as caught:
            run_mission.main(['--duration', '0'])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn('duration must be positive', errors.getvalue())

    def test_animation_rejects_unknown_extension_before_flying(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as caught:
            animate.main(['--output', 'flight.txt'])
        self.assertEqual(caught.exception.code, 1)
        self.assertIn('.mp4 or .gif', errors.getvalue())

    def test_animation_rejects_bad_playback(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as caught:
            animate.main(['--playback', '0'])
        self.assertEqual(caught.exception.code, 1)
        self.assertIn('playback', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
