import unittest
from .utils import *

class UtilsTestCase(unittest.TestCase):

    OUTPUT_DIR = Path("data/test_output")
    # https://ccmixter.org/files/admiralbob77/65751
    SAMPLE_MP3 = Path("data/admiralbob77_-_Creator_of_the_Stars_At_Night_3.mp3")


    @classmethod
    def setUpClass(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if cls.OUTPUT_DIR.exists():
            cls.OUTPUT_DIR.rmdir()

    # @unittest.skip("Skip testDemucs")
    def testDemucs(self):
        separate_vocals(
            filepath=[self.SAMPLE_MP3],
            output_dir=self.OUTPUT_DIR,
            should_speedup=False,
            shifts=1,
            parallel=1,
        )
        filename = self.SAMPLE_MP3.stem
        output_dir = Path(f"{self.OUTPUT_DIR}/{filename}")
        vocals_path = Path(f"{output_dir}/vocals.wav")
        no_vocals_path = Path(f"{output_dir}/no_vocals.wav")
        self.assertTrue(vocals_path.exists())
        self.assertTrue(no_vocals_path.exists())
        no_vocals_path.unlink()
        return vocals_path

    def testVadCut(self):
        vocals_path = self.testDemucs()
        filename = self.SAMPLE_MP3.stem
        output_dir = Path(f"{self.OUTPUT_DIR}/{filename}/cut")
        pipeline = get_voice_activity_segments()
        vad_cuts = vad_cut(
            pipeline=pipeline,
            audio_filepath=vocals_path,
            output_dir=output_dir,
        )
        self.assertTrue(output_dir.exists())
        for cut in vad_cuts:
            self.assertTrue(cut.filepath.exists())
            cut.filepath.unlink()
        output_dir.rmdir()
        vocals_path.unlink()
        output_dir.parent.rmdir()


if __name__ == "__main__":
    unittest.main()