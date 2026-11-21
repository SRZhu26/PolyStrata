"""The vision section's "max_size" (opt-in): a picture larger than the box either way round is scaled down to fit it
before the encoder reads it; one that fits, and every picture when the key is not set, goes through as before."""
import io
import unittest

from serve.server import Vision

try:
    from PIL import Image
except ImportError:                 # Pillow is optional: the image then goes through as it was
    Image = None

BOX = (2560, 1440)


def picture(size, fmt="PNG", orientation=None) -> bytes:
    im = Image.new("RGB", size, (30, 120, 200))
    out = io.BytesIO()
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        im.save(out, format=fmt, exif=exif)
    else:
        im.save(out, format=fmt)
    return out.getvalue()


def size_of(data: bytes):
    return Image.open(io.BytesIO(data)).size


class Fit(unittest.TestCase):
    def test_the_box_either_way_round(self):
        self.assertEqual(Vision.fit((3840, 2160), BOX), (2560, 1440))      # 4K landscape
        self.assertEqual(Vision.fit((2160, 3840), BOX), (1440, 2560))      # 4K portrait
        self.assertEqual(Vision.fit((1440, 3200), BOX), (1152, 2560))      # tall phone screenshot: the long side
        self.assertEqual(Vision.fit((3440, 1440), BOX), (2560, 1072))      # ultrawide: the long side
        self.assertEqual(Vision.fit((2000, 2000), BOX), (1440, 1440))      # square: the short side

    def test_a_picture_that_fits_is_left_alone(self):
        for size in ((2560, 1440), (1440, 2560), (1920, 1080), (40, 20), (2560, 100)):
            self.assertIsNone(Vision.fit(size, BOX), size)

    def test_no_box(self):
        self.assertIsNone(Vision.fit((9000, 9000), None))

    def test_the_setting(self):
        self.assertEqual(Vision.max_size_of([2560, 1440]), BOX)
        self.assertEqual(Vision.max_size_of("2560x1440"), BOX)
        for unset in (None, "", 0):
            self.assertIsNone(Vision.max_size_of(unset))
        for bad in ("1440p", [2560], [0, 1440], "axb"):
            with self.assertRaises(ValueError):
                Vision.max_size_of(bad)


@unittest.skipIf(Image is None, "Pillow is not installed")
class Normalize(unittest.TestCase):
    def test_larger_pictures_come_out_scaled_down_as_png(self):
        for fmt in ("PNG", "JPEG", "BMP"):
            out = Vision.normalize(picture((3840, 2160), fmt), BOX)
            self.assertEqual(out[:8], b"\x89PNG\r\n\x1a\n", fmt)
            self.assertEqual(size_of(out), (2560, 1440), fmt)

    def test_a_turned_phone_photo_is_turned_upright_and_scaled(self):
        out = Vision.normalize(picture((4000, 3000), "JPEG", orientation=6), BOX)
        self.assertEqual(size_of(out), (1440, 1920))

    def test_a_picture_that_fits_passes_through_unchanged(self):
        for fmt in ("PNG", "JPEG"):
            data = picture((2560, 1440), fmt)
            self.assertIs(Vision.normalize(data, BOX), data, fmt)

    def test_without_the_setting_nothing_changes(self):
        data = picture((3840, 2160), "PNG")
        self.assertIs(Vision.normalize(data), data)
        self.assertIs(Vision.normalize(data, None), data)


if __name__ == "__main__":
    unittest.main()
