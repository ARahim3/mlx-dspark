"""Race verdict logic — model-free (the comparison is a static method over token ids)."""

from mlx_dspark.server import Engine


def arm(index, label, token_ids):
    return {"index": index, "label": label, "token_ids": token_ids, "text": ""}


class TestVerdict:
    def test_identical_arms(self):
        v = Engine._race_verdict([arm(0, "dspark", [1, 2, 3]), arm(1, "baseline", [1, 2, 3])])
        assert v["identical"] is True
        assert v["divergences"] == []

    def test_needs_two_arms(self):
        v = Engine._race_verdict([arm(0, "dspark", [1, 2])])
        assert v["comparable"] is False

    def test_reports_first_divergence_and_both_tokens(self):
        v = Engine._race_verdict([arm(0, "dspark", [1, 2, 9, 4]),
                                  arm(1, "baseline", [1, 2, 7, 4])])
        assert v["identical"] is False
        d = v["divergences"][0]
        assert d["first_diff"] == 2
        assert d["reference_token"] == 9
        assert d["arm_token"] == 7
        assert d["length_only"] is False

    def test_length_difference_is_not_a_divergence(self):
        """A speculative arm commits a whole block, so it can overshoot the token limit. The
        arms agree everywhere they overlap — calling that 'diverged' would cry wolf on nearly
        every race and make the real signal worthless."""
        v = Engine._race_verdict([arm(0, "dspark", [1, 2, 3, 4, 5]),
                                  arm(1, "baseline", [1, 2, 3])])
        assert v["identical"] is True
        assert v["divergences"][0]["length_only"] is True
        assert v["divergences"][0]["first_diff"] == 3

    def test_content_divergence_wins_over_length(self):
        v = Engine._race_verdict([arm(0, "a", [1, 2, 3, 4, 5]), arm(1, "b", [1, 9])])
        d = v["divergences"][0]
        assert d["first_diff"] == 1 and d["length_only"] is False
        assert v["identical"] is False

    def test_compares_every_arm_against_the_first(self):
        v = Engine._race_verdict([arm(0, "ref", [1, 2, 3]),
                                  arm(1, "b", [1, 2, 3]),
                                  arm(2, "c", [1, 5, 3])])
        assert v["reference"] == "ref"
        assert [d["arm"] for d in v["divergences"]] == [2]

    def test_empty_arm_is_reported_not_crashed(self):
        v = Engine._race_verdict([arm(0, "ref", [1, 2]), arm(1, "empty", [])])
        assert v["divergences"][0]["first_diff"] == 0


class TestVerdictWording:
    """The verdict is the app's headline claim, so its wording is behaviour, not decoration."""

    def test_identical_says_so_plainly(self):
        v = Engine._race_verdict([arm(0, "a", [1, 2]), arm(1, "b", [1, 2])])
        assert "same tokens" in Engine._verdict_detail(v)

    def test_length_only_is_not_described_as_a_disagreement(self):
        v = Engine._race_verdict([arm(0, "a", [1, 2, 3]), arm(1, "b", [1, 2])])
        detail = Engine._verdict_detail(v)
        assert "where they overlap" in detail
        assert "diverged" not in detail.lower()

    def test_real_divergence_is_not_reported_as_a_fault(self):
        """Probed on-device 2026-07-22: at a real divergence every forward path agreed on the
        ranking — the arms parted because their KV caches drifted, not because an accept rule
        misfired. The wording must not send anyone hunting a bug that isn't there."""
        v = Engine._race_verdict([arm(0, "dspark", [1, 2, 9]), arm(1, "baseline", [1, 2, 7])])
        detail = Engine._verdict_detail(v)
        assert "equally valid" in detail
        assert "top choice" in detail
        for alarming in ("wrong", "should not happen", "investigating", "bug"):
            assert alarming not in detail.lower()
