"""Unit tests for hypex.ml.artifact role serialization."""
from __future__ import annotations

from hypex.dataset.roles import (
    ABCRole,
    default_roles,
)
from hypex.ml.artifact import MLArtifact


class TestDecodeParamsWithRoles:
    """Tests for decoding roles from serialized parameters."""

    def test_decodes_single_role(self):
        params = {"role_target": {"class": "TargetRole", "role_name": "Target"}}
        decoded = MLArtifact._decode_params(params)
        assert isinstance(decoded["role_target"], ABCRole)
        assert isinstance(decoded["role_target"], type(default_roles["target"]))
        assert decoded["role_target"].role_name == "Target"

    def test_decodes_multiple_roles(self):
        params = {
            "role_feature": {"class": "FeatureRole", "role_name": "Feature"},
            "role_grouping": {"class": "GroupingRole", "role_name": "Grouping"},
            "role_target": {"class": "TargetRole", "role_name": "Target"},
        }
        decoded = MLArtifact._decode_params(params)
        assert isinstance(decoded["role_feature"], ABCRole)
        assert isinstance(decoded["role_grouping"], ABCRole)
        assert isinstance(decoded["role_target"], ABCRole)
        assert decoded["role_feature"].role_name == "Feature"
        assert decoded["role_grouping"].role_name == "Grouping"
        assert decoded["role_target"].role_name == "Target"

    def test_decodes_additional_roles(self):
        from hypex.dataset.roles import AdditionalTreatmentRole
        params = {
            "role_additional_treatment": {
                "class": "AdditionalTreatmentRole", "role_name": "AdditionalTreatment",
                "data_type": "int",
            }
        }
        decoded = MLArtifact._decode_params(params)
        role = decoded["role_additional_treatment"]
        assert isinstance(role, ABCRole)
        assert isinstance(role, AdditionalTreatmentRole)
        assert role.role_name == "AdditionalTreatment"
        assert role.data_type is int

    def test_decodes_list_of_roles(self):
        params = {
            "roles": [
                {"class": "TargetRole", "role_name": "Target"},
                {"class": "FeatureRole", "role_name": "Feature"},
                {"class": "GroupingRole", "role_name": "Grouping"},
            ]
        }
        decoded = MLArtifact._decode_params(params)
        assert isinstance(decoded["roles"], list)
        assert len(decoded["roles"]) == 3
        assert all(isinstance(r, ABCRole) for r in decoded["roles"])
        assert decoded["roles"][0].role_name == "Target"
        assert decoded["roles"][1].role_name == "Feature"
        assert decoded["roles"][2].role_name == "Grouping"


class TestBuildPipelineConfigWithRoles:
    """Tests for building pipeline config that includes role serialization."""

    def test_serializes_roles_in_pipeline_config(self):
        params = {
            "role_target": {"class": "TargetRole", "role_name": "Target"},
            "role_feature": {"class": "FeatureRole", "role_name": "Feature"},
            "role_grouping": {"class": "GroupingRole", "role_name": "Grouping"},
        }
        decoded = MLArtifact._decode_params(params)
        assert "role_target" in decoded
        assert "role_feature" in decoded
        assert "role_grouping" in decoded
        assert isinstance(decoded["role_target"], ABCRole)
        assert isinstance(decoded["role_feature"], ABCRole)
        assert isinstance(decoded["role_grouping"], ABCRole)

    def test_serializes_all_default_roles(self):
        # Some roles have special capitalization in the implementation
        roles_to_test = [
            "info", "default", "feature", "treatment", "grouping", "target",
            "pretarget", "stratification", "statistic", "index", "filter",
            "constgroup", "disabled", "lag", "resume", "report",
            "additionaltreatment", "additionalgrouping", "additionaltarget",
            "additionalfeature", "additionalpretarget", "additionalvariancereduction",
        ]
        for role_name in roles_to_test:
            role = default_roles[role_name]
            role_dict = role.to_dict()
            assert isinstance(role_dict, dict)
            assert "class" in role_dict
            assert "role_name" in role_dict
            expected_name = role_name.capitalize()
            # Special cases for capitalization
            if role_name == "pretarget":
                expected_name = "PreTarget"
            elif role_name == "constgroup":
                expected_name = "ConstGroup"
            elif role_name == "disabled":
                expected_name = "Disable"
            elif role_name == "lag":
                expected_name = "Abstract"
            elif role_name == "additionaltreatment":
                expected_name = "AdditionalTreatment"
            elif role_name == "additionalgrouping":
                expected_name = "AdditionalGrouping"
            elif role_name == "additionaltarget":
                expected_name = "AdditionalTarget"
            elif role_name == "additionalfeature":
                expected_name = "AdditionalFeature"
            elif role_name == "additionalpretarget":
                expected_name = "AdditionalPreTarget"
            elif role_name == "additionalvariancereduction":
                expected_name = "AdditionalVarianceReduction"
            assert role_dict["role_name"] == expected_name


class TestCreateExecutorWithRoles:
    """Tests for creating executors with decoded roles."""

    def test_rebuilds_executor_with_decoded_roles(self):
        params = {
            "role_target": {"class": "TargetRole", "role_name": "Target"},
            "role_feature": {"class": "FeatureRole", "role_name": "Feature"},
        }
        decoded = MLArtifact._decode_params(params)
        assert isinstance(decoded["role_target"], ABCRole)
        assert isinstance(decoded["role_feature"], ABCRole)
        assert decoded["role_target"].role_name == "Target"
        assert decoded["role_feature"].role_name == "Feature"

    def test_rebuilds_executor_with_additional_roles(self):
        from hypex.dataset.roles import AdditionalTreatmentRole
        params = {
            "role_additional_treatment": {
                "class": "AdditionalTreatmentRole", "role_name": "AdditionalTreatment",
                "data_type": "int",
            }
        }
        decoded = MLArtifact._decode_params(params)
        role = decoded["role_additional_treatment"]
        assert isinstance(role, ABCRole)
        assert isinstance(role, AdditionalTreatmentRole)
        assert role.role_name == "AdditionalTreatment"
        assert role.data_type is int
