"""回归测试：确保对齐、分类、裁决三层行为稳定。

离线组件是确定性的，因此这些断言可以写死——这正是保留 HeuristicReasoner
的价值：它是整条流水线的基线，换成 LLM 后可用同一批用例做对照。
"""
import json, os, subprocess, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guji_collator.normalize import normalize, orthographic_equal
from guji_collator.align import collate, DivType
from guji_collator.ocr import fuse_confidence


class TestNormalize(unittest.TestCase):
    def test_variant_and_simplified_fold(self):
        # 异体、简繁差异必须被消解，不得产生假异文
        for a, b in [("逺", "远"), ("學而時習", "学而时习"), ("汎愛衆", "泛爱众")]:
            self.assertTrue(orthographic_equal(a, b), f"{a} vs {b}")

    def test_index_map_roundtrip(self):
        n = normalize("学而时习之，不亦悦乎？")
        self.assertNotIn("，", n.cnf)
        self.assertEqual(n.raw_slice(0, 2), "学而")
        self.assertIn(4, n.punct_after)      # 「之」后有标点


class TestCollate(unittest.TestCase):
    def setUp(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        b = open(os.path.join(root, "data/demo/base_shanben_ocr.txt"), encoding="utf-8").read()
        m = open(os.path.join(root, "data/demo/modern_dianjiao_ocr.txt"), encoding="utf-8").read()
        self.divs = collate(normalize(b), normalize(m))
        self.types = {d.base_text: d.dtype for d in self.divs}

    def test_no_false_positives(self):
        # 汎/泛、逺/远、衆/众 等纯字形差异不得出现在异文表里
        for ch in ("汎", "逺", "衆", "爲"):
            self.assertNotIn(ch, self.types, f"{ch} 被误判为异文")

    def test_classification(self):
        self.assertIs(self.types["說"], DivType.PHONETIC_LOAN)
        self.assertIs(self.types["□"], DivType.LACUNA)
        self.assertIs(self.types["樂"], DivType.DITTOGRAPHY)
        self.assertIs(self.types["眚"], DivType.OCR_SHAPE)
        self.assertIs(self.types["謹"], DivType.SUBSTANTIVE)
        self.assertIs(self.types["文學"], DivType.TRANSPOSITION)

    def test_taboo_detected(self):
        self.assertIs(self.types["人"], DivType.TABOO)


class TestFusion(unittest.TestCase):
    def test_evidence_accumulates(self):
        self.assertGreater(fuse_confidence(0.8, 0.8), 0.8)
        self.assertLess(fuse_confidence(0.05, 0.9), 0.5)


class TestEndToEnd(unittest.TestCase):
    def test_run_and_apply(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r = subprocess.run([sys.executable, "-m", "guji_collator", "run",
                            "--task", "data/demo/task.json"],
                           cwd=root, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        rep = json.load(open(os.path.join(root, "outputs/lunyu-xueer/run_report.json"),
                             encoding="utf-8"))
        self.assertEqual(rep["stats"]["total"], 11)
        self.assertGreaterEqual(rep["stats"]["suspended"], 1)   # 必须有敢于悬置的行为
        # 低置信 OCR 位置一定进入悬置，绝不能被自动改字
        for v in rep["verdicts"]:
            if v["dtype"] == "形近疑誤" and not v["suspended"]:
                self.fail("形近疑误未经人工确认即自动定案")
        # 公开本句读不得与今本高度雷同
        self.assertLess(rep["punctuation"]["jaccard"], 0.92)

    def test_public_edition_has_no_modern_punctuation_copy(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        txt = open(os.path.join(root, "outputs/lunyu-xueer/public/critical_edition.md"),
                   encoding="utf-8").read()
        self.assertNotIn("简", txt)
        self.assertIn("底本為公有領域善本", txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
