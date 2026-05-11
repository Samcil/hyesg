"""Tests for hyesg.core.registry."""

from __future__ import annotations

import pytest

from hyesg.core.registry import (
    clear_registry,
    get_model,
    list_models,
    register_model,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure registry is clean before and after each test."""
    clear_registry()
    yield
    clear_registry()


class TestRegisterModel:
    def test_register_and_retrieve(self) -> None:
        @register_model("test_model")
        class TestModel:
            pass

        assert get_model("test_model") is TestModel

    def test_duplicate_registration_raises(self) -> None:
        @register_model("dup")
        class First:
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_model("dup")
            class Second:
                pass

    def test_register_preserves_class(self) -> None:
        """Decorator should return the original class."""

        @register_model("preserved")
        class MyModel:
            x = 42

        assert MyModel.x == 42


class TestGetModel:
    def test_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown model"):
            get_model("nonexistent")

    def test_error_message_lists_available(self) -> None:
        @register_model("alpha")
        class Alpha:
            pass

        @register_model("beta")
        class Beta:
            pass

        with pytest.raises(KeyError) as exc_info:
            get_model("gamma")
        assert "alpha" in str(exc_info.value)
        assert "beta" in str(exc_info.value)


class TestListModels:
    def test_empty_registry(self) -> None:
        assert list_models() == []

    def test_lists_sorted(self) -> None:
        @register_model("zebra")
        class Z:
            pass

        @register_model("alpha")
        class A:
            pass

        assert list_models() == ["alpha", "zebra"]


class TestClearRegistry:
    def test_clear(self) -> None:
        @register_model("temp")
        class Temp:
            pass

        assert len(list_models()) == 1
        clear_registry()
        assert len(list_models()) == 0
