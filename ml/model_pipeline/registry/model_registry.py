from __future__ import annotations

from typing import List, Optional, Dict, Any

from mlflow.tracking import MlflowClient


class ModelRegistryManager:
    """
    Small helper around the MLflow Model Registry to:
      - list versions & stages
      - promote a version to Staging / Production
      - rollback Production to a previous version


    """

    def __init__(self, tracking_uri: str = "mlruns"):
        # Use the same tracking URI as the rest of our project
        self.client = MlflowClient(tracking_uri=tracking_uri)

    # ------------------------------------------------------------------
    # List all versions + stages
    # ------------------------------------------------------------------
    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Return a list of all versions for a given registered model
        with their current stage and some extra metadata.
        """
        versions = self.client.search_model_versions(f"name='{model_name}'")
        out: List[Dict[str, Any]] = []
        for v in versions:
            out.append(
                {
                    "version": int(v.version),
                    "current_stage": v.current_stage,
                    "run_id": v.run_id,
                    "creation_timestamp": getattr(v, "creation_timestamp", None),
                    "last_updated_timestamp": getattr(v, "last_updated_timestamp", None),
                }
            )

        # Sort by version number for readability
        out.sort(key=lambda x: x["version"])
        return out

    # ------------------------------------------------------------------
    # Generic promotion helper
    # ------------------------------------------------------------------
    def _transition_to_stage(
        self,
        model_name: str,
        version: int,
        new_stage: str,
    ) -> None:
        """
        Internal helper that changes the stage of a single model version.
        We do NOT pass archive_existing here (not supported in some MLflow versions).
        Any archiving of previous versions is handled manually in the
        higher-level methods.
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage=new_stage,
        )

    # ------------------------------------------------------------------
    # Promote to STAGING
    # ------------------------------------------------------------------
    def promote_to_staging(self, model_name: str, version: int) -> Dict[str, Any]:
        """
        Promote a given version to STAGING.

        Note:
        - This does NOT automatically demote other STAGING versions.
        """
        self._transition_to_stage(model_name, version, "Staging")
        return {
            "success": True,
            "model_name": model_name,
            "version": int(version),
            "new_stage": "Staging",
        }

    # ------------------------------------------------------------------
    # Promote to PRODUCTION (with manual archiving of old Production)
    # ------------------------------------------------------------------
    def promote_to_production(self, model_name: str, version: int) -> Dict[str, Any]:
        """
        Promote a given version to PRODUCTION.

        We manually:
          - Find any existing Production versions and move them to 'Archived'
          - Promote the requested version to 'Production'
        """
        # 1) Archive existing Production versions (if any)
        existing_prod = self.client.search_model_versions(
            f"name='{model_name}' and current_stage='Production'"
        )
        for v in existing_prod:
            if int(v.version) != int(version):
                self._transition_to_stage(model_name, int(v.version), "Archived")

        # 2) Promote the new Production version
        self._transition_to_stage(model_name, version, "Production")

        return {
            "success": True,
            "model_name": model_name,
            "version": int(version),
            "new_stage": "Production",
            "archived_previous_production_versions": [int(v.version) for v in existing_prod],
        }

    # ------------------------------------------------------------------
    # Rollback PRODUCTION to a previous version
    # ------------------------------------------------------------------
    def rollback_to_version(self, model_name: str, target_version: int) -> Dict[str, Any]:
        """
        Rollback the Production model to a specific *existing* version.

        Steps:
          - Ensure target_version exists
          - Archive current Production versions (if any)
          - Promote target_version to Production
        """
        # Ensure the target version actually exists
        try:
            self.client.get_model_version(name=model_name, version=str(target_version))
        except Exception as e:
            return {
                "success": False,
                "error": f"Target version {target_version} does not exist for model '{model_name}': {e}",
            }

        # 1) Archive current Production version(s)
        current_prod = self.client.search_model_versions(
            f"name='{model_name}' and current_stage='Production'"
        )
        archived_versions: List[int] = []
        for v in current_prod:
            if int(v.version) != int(target_version):
                self._transition_to_stage(model_name, int(v.version), "Archived")
                archived_versions.append(int(v.version))

        # 2) Promote the target version to Production
        self._transition_to_stage(model_name, int(target_version), "Production")

        return {
            "success": True,
            "model_name": model_name,
            "reinstated_version": int(target_version),
            "archived_previous_production_versions": archived_versions,
        }

    # ------------------------------------------------------------------
    # Get current Production model version
    # ------------------------------------------------------------------
    def get_current_production(self, model_name: str) -> Optional[int]:
        """
        Return the version number of the latest model in PRODUCTION,
        or None if there is no Production model yet.
        """
        versions = self.client.get_latest_versions(model_name, stages=["Production"])
        return int(versions[0].version) if versions else None