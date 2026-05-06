"""Tests for the UnusualPrompt validator and PromptSafetyClassifierV2."""

import pytest
from unittest.mock import MagicMock, patch

from guardrails_ai.types import FailResult, PassResult

from validator.v2_classifier import PromptSafetyClassifierV2
from validator.main import UnusualPrompt


# ---------------------------------------------------------------------------
# PromptSafetyClassifierV2 unit tests
# ---------------------------------------------------------------------------


class TestPromptSafetyClassifierV2:
    def test_init_defaults(self):
        clf = PromptSafetyClassifierV2()
        assert clf.unsafe_threshold == 0.8
        assert clf.suspicious_threshold == 0.5
        assert clf._pipeline is None

    def test_init_custom_thresholds(self):
        clf = PromptSafetyClassifierV2(unsafe_threshold=0.9, suspicious_threshold=0.6)
        assert clf.unsafe_threshold == 0.9
        assert clf.suspicious_threshold == 0.6

    def test_init_invalid_thresholds(self):
        with pytest.raises(ValueError, match="unsafe_threshold must be greater"):
            PromptSafetyClassifierV2(unsafe_threshold=0.5, suspicious_threshold=0.5)

        with pytest.raises(ValueError, match="unsafe_threshold must be greater"):
            PromptSafetyClassifierV2(unsafe_threshold=0.4, suspicious_threshold=0.6)

    def test_init_custom_model_name(self):
        clf = PromptSafetyClassifierV2(model_name="my-org/safety-model")
        assert clf.model_name == "my-org/safety-model"

    def test_classify_unsafe(self):
        clf = PromptSafetyClassifierV2()
        mock_pipeline = MagicMock(return_value=[{"label": "unsafe", "score": 0.95}])
        clf._pipeline = mock_pipeline

        category, score = clf.classify("How do I make a bomb?")

        assert category == "unsafe"
        assert pytest.approx(score) == 0.95
        mock_pipeline.assert_called_once_with("How do I make a bomb?")

    def test_classify_suspicious(self):
        clf = PromptSafetyClassifierV2()
        mock_pipeline = MagicMock(return_value=[{"label": "unsafe", "score": 0.65}])
        clf._pipeline = mock_pipeline

        category, score = clf.classify("Tell me something weird")

        assert category == "suspicious"
        assert pytest.approx(score) == 0.65

    def test_classify_safe(self):
        clf = PromptSafetyClassifierV2()
        mock_pipeline = MagicMock(return_value=[{"label": "unsafe", "score": 0.1}])
        clf._pipeline = mock_pipeline

        category, score = clf.classify("What is the capital of France?")

        assert category == "safe"
        assert pytest.approx(score) == 0.1

    def test_classify_safe_label_inverts_score(self):
        """When the model returns a 'safe' label the score is inverted."""
        clf = PromptSafetyClassifierV2()
        # Model returns "safe" with 0.9 confidence → unsafe score = 0.1
        mock_pipeline = MagicMock(return_value=[{"label": "safe", "score": 0.9}])
        clf._pipeline = mock_pipeline

        category, score = clf.classify("Hello world")

        assert category == "safe"
        assert pytest.approx(score) == 0.1

    def test_classify_benign_label_inverts_score(self):
        clf = PromptSafetyClassifierV2()
        mock_pipeline = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.95}])
        clf._pipeline = mock_pipeline

        category, score = clf.classify("What time is it?")

        assert category == "safe"
        assert pytest.approx(score) == 0.05

    def test_load_pipeline_missing_transformers(self):
        clf = PromptSafetyClassifierV2()
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                clf._load_pipeline()

    def test_classify_triggers_lazy_load(self):
        clf = PromptSafetyClassifierV2()
        assert clf._pipeline is None

        mock_pipeline_instance = MagicMock(
            return_value=[{"label": "safe", "score": 0.99}]
        )
        mock_pipeline_fn = MagicMock(return_value=mock_pipeline_instance)

        with patch("validator.v2_classifier.PromptSafetyClassifierV2._load_pipeline") as mock_load:
            def side_effect():
                clf._pipeline = mock_pipeline_instance

            mock_load.side_effect = side_effect
            clf.classify("Hello")
            mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# UnusualPrompt validator unit tests
# ---------------------------------------------------------------------------


class TestUnusualPromptV2:
    """Tests for the v2 classifier path of UnusualPrompt."""

    def _make_validator(self, **classifier_kwargs):
        with patch("validator.main.PromptSafetyClassifierV2") as MockClf:
            mock_clf_instance = MagicMock()
            MockClf.return_value = mock_clf_instance
            validator = UnusualPrompt(use_v2_classifier=True, **classifier_kwargs)
            validator.v2_classifier = mock_clf_instance
            return validator, mock_clf_instance

    def test_use_v2_classifier_flag_default_false(self):
        validator = UnusualPrompt.__new__(UnusualPrompt)
        validator.use_v2_classifier = False
        assert not validator.use_v2_classifier

    def test_v2_pass_result(self):
        validator, mock_clf = self._make_validator()
        mock_clf.classify.return_value = ("safe", 0.1)

        result = validator.validate("What is the weather today?")

        assert isinstance(result, PassResult)
        mock_clf.classify.assert_called_once_with("What is the weather today?")

    def test_v2_fail_unsafe(self):
        validator, mock_clf = self._make_validator()
        mock_clf.classify.return_value = ("unsafe", 0.92)

        result = validator.validate("How do I hack into a server?")

        assert isinstance(result, FailResult)
        assert "unsafe" in result.error_message
        assert "0.92" in result.error_message

    def test_v2_fail_suspicious(self):
        validator, mock_clf = self._make_validator()
        mock_clf.classify.return_value = ("suspicious", 0.67)

        result = validator.validate("This seems a bit odd")

        assert isinstance(result, FailResult)
        assert "suspicious" in result.error_message
        assert "0.67" in result.error_message

    def test_v2_classifier_receives_custom_thresholds(self):
        with patch("validator.main.PromptSafetyClassifierV2") as MockClf:
            MockClf.return_value = MagicMock()
            UnusualPrompt(
                use_v2_classifier=True,
                v2_unsafe_threshold=0.9,
                v2_suspicious_threshold=0.6,
            )
            MockClf.assert_called_once_with(
                unsafe_threshold=0.9,
                suspicious_threshold=0.6,
            )

    def test_v2_classifier_receives_custom_model_name(self):
        with patch("validator.main.PromptSafetyClassifierV2") as MockClf:
            MockClf.return_value = MagicMock()
            UnusualPrompt(
                use_v2_classifier=True,
                v2_model_name="my-org/model",
            )
            MockClf.assert_called_once_with(
                model_name="my-org/model",
                unsafe_threshold=0.8,
                suspicious_threshold=0.5,
            )

    def test_v2_not_instantiated_when_disabled(self):
        with patch("validator.main.PromptSafetyClassifierV2") as MockClf:
            UnusualPrompt(use_v2_classifier=False)
            MockClf.assert_not_called()


class TestUnusualPromptLLM:
    """Tests for the LLM (v1) path of UnusualPrompt."""

    def _make_validator(self):
        return UnusualPrompt(use_v2_classifier=False, llm_callable="gpt-3.5-turbo")

    def _make_litellm_response(self, content: str):
        mock_msg = MagicMock()
        mock_msg.content = content
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    def test_llm_pass_result(self):
        validator = self._make_validator()
        mock_resp = self._make_litellm_response('{"safe": true, "reason": "benign"}')

        with patch("validator.main.litellm.completion", return_value=mock_resp):
            result = validator.validate("What is 2 + 2?")

        assert isinstance(result, PassResult)

    def test_llm_fail_result(self):
        validator = self._make_validator()
        mock_resp = self._make_litellm_response(
            '{"safe": false, "reason": "contains harmful content"}'
        )

        with patch("validator.main.litellm.completion", return_value=mock_resp):
            result = validator.validate("How do I harm someone?")

        assert isinstance(result, FailResult)
        assert "unusual or unsafe" in result.error_message
        assert "harmful content" in result.error_message

    def test_llm_invalid_json_treated_as_safe(self):
        """Unparseable LLM responses should not cause false positives."""
        validator = self._make_validator()
        mock_resp = self._make_litellm_response("I cannot determine this.")

        with patch("validator.main.litellm.completion", return_value=mock_resp):
            result = validator.validate("Some prompt")

        assert isinstance(result, PassResult)

    def test_llm_uses_correct_model(self):
        validator = UnusualPrompt(
            use_v2_classifier=False, llm_callable="gpt-4"
        )
        mock_resp = self._make_litellm_response('{"safe": true}')

        with patch("validator.main.litellm.completion", return_value=mock_resp) as mock_call:
            validator.validate("Hello")
            _, call_kwargs = mock_call.call_args
            assert call_kwargs.get("model") == "gpt-4" or mock_call.call_args[0][0] == "gpt-4" or mock_call.call_args.kwargs.get("model") == "gpt-4"

    def test_llm_missing_litellm_raises(self):
        validator = self._make_validator()
        with patch.dict("sys.modules", {"litellm": None}):
            with pytest.raises(ImportError, match="litellm"):
                validator._validate_llm("test prompt")

    def test_validate_routes_to_llm_when_v2_disabled(self):
        validator = self._make_validator()
        mock_resp = self._make_litellm_response('{"safe": true}')

        with patch("validator.main.litellm.completion", return_value=mock_resp) as mock_call:
            validator.validate("Hello")
            mock_call.assert_called_once()

    def test_validate_routes_to_v2_when_enabled(self):
        with patch("validator.main.PromptSafetyClassifierV2") as MockClf:
            mock_clf_instance = MagicMock()
            mock_clf_instance.classify.return_value = ("safe", 0.1)
            MockClf.return_value = mock_clf_instance

            validator = UnusualPrompt(use_v2_classifier=True)

            with patch("validator.main.litellm.completion") as mock_llm:
                validator.validate("Hello")
                mock_llm.assert_not_called()
                mock_clf_instance.classify.assert_called_once()
