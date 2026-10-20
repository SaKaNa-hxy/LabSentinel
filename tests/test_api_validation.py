import asyncio
import io
import unittest

from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from app.backend.main import MAX_IMAGE_BYTES, analyze


class ApiValidationTests(unittest.TestCase):
    def test_oversized_image_is_rejected_before_model_inference(self):
        upload = UploadFile(
            file=io.BytesIO(b"x" * (MAX_IMAGE_BYTES + 1)),
            filename="oversized.jpg",
            headers=Headers({"content-type": "image/jpeg"}),
        )
        with self.assertRaises(HTTPException) as result:
            asyncio.run(analyze(upload))
        self.assertEqual(result.exception.status_code, 413)


if __name__ == "__main__":
    unittest.main()
