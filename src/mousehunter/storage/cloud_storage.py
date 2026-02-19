"""
Cloud Storage Module - rclone-based cloud archival for MouseHunter.

Provides reliable cloud uploads via rclone with:
- Automatic metadata generation per detection
- Upload tracking to avoid duplicates
- Local and cloud retention policies
- Background sync capabilities

Supported providers: Any rclone-supported remote (Google Drive, Dropbox, S3, etc.)
"""

import asyncio
import json
import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Check if rclone is available
RCLONE_AVAILABLE = shutil.which("rclone") is not None


class CloudStorage:
    """
    Cloud storage manager using rclone for reliable uploads.

    Handles:
    - Upload evidence folders to cloud with metadata
    - Track upload status to avoid duplicates
    - Enforce local retention (delete after upload)
    - Periodic sync of evidence directory
    """

    def __init__(
        self,
        rclone_remote: str,
        remote_path: str = "MouseHunter",
        evidence_dir: str = "runtime/evidence",
        enabled: bool = True,
        delete_local_after_upload: bool = False,
        retention_days_local: int = 7,
        retention_days_cloud: int = 365,
        bandwidth_limit: str = "",
    ):
        """
        Initialize cloud storage manager.

        Args:
            rclone_remote: rclone remote name (e.g., 'gdrive', 'dropbox')
            remote_path: Base path on remote storage
            evidence_dir: Local evidence directory path
            enabled: Enable cloud uploads
            delete_local_after_upload: Delete local files after successful upload
            retention_days_local: Days to keep local files (0 for forever)
            retention_days_cloud: Days to keep cloud files (0 for forever)
            bandwidth_limit: Bandwidth limit (e.g., '1M' for 1MB/s)
        """
        self.rclone_remote = rclone_remote
        self.remote_path = remote_path
        self.evidence_dir = Path(evidence_dir)
        self.enabled = enabled and RCLONE_AVAILABLE and rclone_remote
        self.delete_local_after_upload = delete_local_after_upload
        self.retention_days_local = retention_days_local
        self.retention_days_cloud = retention_days_cloud
        self.bandwidth_limit = bandwidth_limit

        # Upload tracker file
        self._tracker_file = self.evidence_dir / ".upload_tracker.json"
        self._upload_status: dict[str, dict[str, Any]] = {}
        self._load_tracker()

        # Dedicated thread pool for rclone (isolates from default asyncio executor)
        self._upload_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rclone")

        # Upload guard to prevent stacking
        self._upload_in_progress = False

        # Sync lock to prevent concurrent syncs
        self._sync_lock = asyncio.Lock()

        if not RCLONE_AVAILABLE:
            logger.warning("rclone not found in PATH - cloud storage disabled")
        elif not rclone_remote:
            logger.warning("No rclone remote configured - cloud storage disabled")
        else:
            logger.info(
                f"Cloud storage initialized: {rclone_remote}:{remote_path} "
                f"(enabled={enabled}, delete_after_upload={delete_local_after_upload})"
            )

    def _load_tracker(self) -> None:
        """Load upload tracker from disk."""
        if self._tracker_file.exists():
            try:
                with open(self._tracker_file) as f:
                    self._upload_status = json.load(f)
                logger.debug(f"Loaded upload tracker: {len(self._upload_status)} entries")
            except Exception as e:
                logger.error(f"Failed to load upload tracker: {e}")
                self._upload_status = {}
        else:
            self._upload_status = {}

    def _save_tracker(self) -> None:
        """Save upload tracker to disk."""
        try:
            self._tracker_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._tracker_file, "w") as f:
                json.dump(self._upload_status, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save upload tracker: {e}")

    def _run_rclone(self, args: list[str], timeout: int = 120) -> tuple[bool, str]:
        """
        Run an rclone command.

        Args:
            args: rclone command arguments
            timeout: Command timeout in seconds

        Returns:
            Tuple of (success, output/error message)
        """
        cmd = ["rclone"] + args

        # Add bandwidth limit if configured
        if self.bandwidth_limit:
            cmd.extend(["--bwlimit", self.bandwidth_limit])

        logger.debug(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout}s"
        except Exception as e:
            return False, str(e)

    def _get_remote_path(self, local_path: Path) -> str:
        """
        Generate remote path with date-based folder structure.

        Local: evidence/prey_20260124_143022/
        Remote: MouseHunter/2026/01/24/prey_20260124_143022/
        """
        folder_name = local_path.name

        # Extract date from folder name (prey_YYYYMMDD_HHMMSS)
        try:
            if folder_name.startswith("prey_"):
                date_str = folder_name.split("_")[1]  # YYYYMMDD
                year = date_str[:4]
                month = date_str[4:6]
                day = date_str[6:8]
                return f"{self.rclone_remote}:{self.remote_path}/{year}/{month}/{day}/{folder_name}"
        except (IndexError, ValueError):
            pass

        # Fallback: use current date
        now = datetime.now()
        return f"{self.rclone_remote}:{self.remote_path}/{now.year}/{now.month:02d}/{now.day:02d}/{folder_name}"

    def create_metadata(
        self,
        evidence_path: Path,
        prey_type: str,
        confidence: float,
        cat_confidence: float | None = None,
        additional_data: dict[str, Any] | None = None,
    ) -> Path:
        """
        Create metadata.json file for a detection.

        Args:
            evidence_path: Path to evidence folder
            prey_type: Detected prey type (e.g., 'rodent')
            confidence: Prey detection confidence
            cat_confidence: Cat detection confidence
            additional_data: Additional metadata fields

        Returns:
            Path to created metadata file
        """
        metadata = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "detection": {
                "prey_type": prey_type,
                "prey_confidence": round(confidence, 4),
                "cat_confidence": round(cat_confidence, 4) if cat_confidence else None,
            },
            "device": {
                "hostname": self._get_hostname(),
                "evidence_path": str(evidence_path),
            },
            "files": [f.name for f in evidence_path.iterdir() if f.is_file()],
        }

        if additional_data:
            metadata["extra"] = additional_data

        metadata_path = evidence_path / "metadata.json"
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Created metadata: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to create metadata: {e}")

        return metadata_path

    def _get_hostname(self) -> str:
        """Get device hostname."""
        try:
            import socket
            return socket.gethostname()
        except Exception:
            return "unknown"

    async def upload_evidence(
        self,
        evidence_path: Path | str,
        prey_type: str = "unknown",
        confidence: float = 0.0,
        cat_confidence: float | None = None,
    ) -> bool:
        """
        Upload an evidence folder to cloud storage.

        Args:
            evidence_path: Path to evidence folder
            prey_type: Detected prey type
            confidence: Detection confidence
            cat_confidence: Cat detection confidence

        Returns:
            True if upload successful
        """
        if not self.enabled:
            logger.debug("Cloud storage disabled, skipping upload")
            return False

        evidence_path = Path(evidence_path)
        if not evidence_path.exists():
            logger.error(f"Evidence path does not exist: {evidence_path}")
            return False

        folder_name = evidence_path.name

        # Check if already uploaded
        if folder_name in self._upload_status:
            status = self._upload_status[folder_name]
            if status.get("uploaded"):
                logger.debug(f"Already uploaded: {folder_name}")
                return True

        # Upload guard: skip if another upload is in progress
        if self._upload_in_progress:
            logger.warning(f"Upload already in progress, deferring: {folder_name}")
            return False

        # Create metadata file
        self.create_metadata(
            evidence_path,
            prey_type=prey_type,
            confidence=confidence,
            cat_confidence=cat_confidence,
        )

        # Generate remote path
        remote_path = self._get_remote_path(evidence_path)

        # Upload using rclone copy (dedicated executor, reduced timeout)
        logger.info(f"Uploading evidence: {folder_name} -> {remote_path}")

        self._upload_in_progress = True
        try:
            success, output = await asyncio.get_running_loop().run_in_executor(
                self._upload_executor,
                lambda: self._run_rclone([
                    "copy",
                    str(evidence_path),
                    remote_path,
                    "--progress",
                ])
            )
        finally:
            self._upload_in_progress = False

        # Update tracker
        self._upload_status[folder_name] = {
            "uploaded": success,
            "upload_time": datetime.now().isoformat(),
            "remote_path": remote_path,
            "error": None if success else output,
        }
        self._save_tracker()

        if success:
            logger.info(f"Upload successful: {folder_name}")

            # Delete local if configured
            if self.delete_local_after_upload:
                try:
                    shutil.rmtree(evidence_path)
                    logger.info(f"Deleted local evidence: {folder_name}")
                except Exception as e:
                    logger.error(f"Failed to delete local evidence: {e}")
        else:
            logger.error(f"Upload failed: {folder_name} - {output}")

        return success

    async def sync_all(self) -> dict[str, bool]:
        """
        Sync all unuploaded evidence folders.

        Returns:
            Dict of folder_name -> upload_success
        """
        if not self.enabled:
            return {}

        async with self._sync_lock:
            results = {}

            if not self.evidence_dir.exists():
                return results

            for folder in self.evidence_dir.iterdir():
                if not folder.is_dir():
                    continue
                if folder.name.startswith("."):
                    continue

                # Check if already uploaded
                if folder.name in self._upload_status:
                    if self._upload_status[folder.name].get("uploaded"):
                        continue

                # Read metadata if exists for prey info
                prey_type = "unknown"
                confidence = 0.0
                metadata_file = folder / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            meta = json.load(f)
                            prey_type = meta.get("detection", {}).get("prey_type", "unknown")
                            confidence = meta.get("detection", {}).get("prey_confidence", 0.0)
                    except Exception as e:
                        logger.debug(f"Metadata parse failed for {folder.name}: {e}")

                success = await self.upload_evidence(
                    folder,
                    prey_type=prey_type,
                    confidence=confidence,
                )
                results[folder.name] = success

            return results

    def cleanup_local(self) -> int:
        """
        Delete local evidence older than retention_days_local.

        Only deletes if:
        - File is older than retention period, OR
        - File has been successfully uploaded (if cloud enabled)

        Returns:
            Number of folders deleted
        """
        if self.retention_days_local <= 0:
            return 0

        deleted = 0
        cutoff = datetime.now() - timedelta(days=self.retention_days_local)

        if not self.evidence_dir.exists():
            return 0

        for folder in self.evidence_dir.iterdir():
            if not folder.is_dir():
                continue
            if folder.name.startswith("."):
                continue

            # Check folder age (use mtime)
            try:
                mtime = datetime.fromtimestamp(folder.stat().st_mtime)
                if mtime > cutoff:
                    continue  # Not old enough
            except Exception:
                continue

            # If cloud enabled, only delete if uploaded
            if self.enabled:
                status = self._upload_status.get(folder.name, {})
                if not status.get("uploaded"):
                    logger.debug(f"Skipping cleanup (not uploaded): {folder.name}")
                    continue

            # Delete folder
            try:
                shutil.rmtree(folder)
                logger.info(f"Cleaned up local evidence: {folder.name}")
                deleted += 1

                # Remove from tracker
                if folder.name in self._upload_status:
                    del self._upload_status[folder.name]

            except Exception as e:
                logger.error(f"Failed to cleanup {folder.name}: {e}")

        if deleted > 0:
            self._save_tracker()

        return deleted

    def get_status(self) -> dict[str, Any]:
        """Get cloud storage status."""
        pending = 0
        uploaded = 0
        failed = 0

        for status in self._upload_status.values():
            if status.get("uploaded"):
                uploaded += 1
            elif status.get("error"):
                failed += 1
            else:
                pending += 1

        # Count local folders not in tracker
        if self.evidence_dir.exists():
            for folder in self.evidence_dir.iterdir():
                if folder.is_dir() and not folder.name.startswith("."):
                    if folder.name not in self._upload_status:
                        pending += 1

        return {
            "enabled": self.enabled,
            "rclone_available": RCLONE_AVAILABLE,
            "rclone_remote": self.rclone_remote,
            "remote_path": self.remote_path,
            "uploaded_count": uploaded,
            "pending_count": pending,
            "failed_count": failed,
            "delete_local_after_upload": self.delete_local_after_upload,
            "retention_days_local": self.retention_days_local,
        }

    def verify_remote(self) -> tuple[bool, str]:
        """
        Verify rclone remote is configured and accessible.

        Returns:
            Tuple of (success, message)
        """
        if not RCLONE_AVAILABLE:
            return False, "rclone not found in PATH"

        if not self.rclone_remote:
            return False, "No rclone remote configured"

        # Check if remote exists
        success, output = self._run_rclone(["listremotes"])
        if not success:
            return False, f"Failed to list remotes: {output}"

        remotes = [r.strip().rstrip(":") for r in output.strip().split("\n") if r.strip()]
        if self.rclone_remote not in remotes:
            return False, f"Remote '{self.rclone_remote}' not found. Available: {remotes}"

        # Try to access remote
        success, output = self._run_rclone([
            "lsd",
            f"{self.rclone_remote}:",
            "--max-depth", "1",
        ], timeout=30)

        if success:
            return True, f"Remote '{self.rclone_remote}' is accessible"
        else:
            return False, f"Cannot access remote: {output}"


# Global instance
_cloud_storage_instance: CloudStorage | None = None


def _create_default_cloud_storage() -> CloudStorage:
    """Create cloud storage from config."""
    try:
        from mousehunter.config import cloud_storage_config, recording_config

        return CloudStorage(
            rclone_remote=cloud_storage_config.rclone_remote,
            remote_path=cloud_storage_config.remote_path,
            evidence_dir=recording_config.evidence_dir,
            enabled=cloud_storage_config.enabled,
            delete_local_after_upload=cloud_storage_config.delete_local_after_upload,
            retention_days_local=recording_config.max_age_days,
            retention_days_cloud=cloud_storage_config.retention_days_cloud,
            bandwidth_limit=cloud_storage_config.bandwidth_limit,
        )
    except ImportError:
        logger.warning("Config not available, using defaults")
        return CloudStorage(rclone_remote="", enabled=False)


def get_cloud_storage() -> CloudStorage:
    """Get or create the global cloud storage instance."""
    global _cloud_storage_instance
    if _cloud_storage_instance is None:
        _cloud_storage_instance = _create_default_cloud_storage()
    return _cloud_storage_instance


async def test_cloud_storage() -> None:
    """Test cloud storage configuration (CLI entry point)."""
    logging.basicConfig(level=logging.INFO)
    print("=== Cloud Storage Test ===")
    print(f"rclone available: {RCLONE_AVAILABLE}")

    # Load from config
    storage = _create_default_cloud_storage()

    print(f"\nConfiguration:")
    print(f"  Enabled: {storage.enabled}")
    print(f"  Remote: {storage.rclone_remote}")
    print(f"  Remote path: {storage.remote_path}")
    print(f"  Evidence dir: {storage.evidence_dir}")
    print(f"  Delete after upload: {storage.delete_local_after_upload}")
    print(f"  Local retention: {storage.retention_days_local} days")
    print(f"  Cloud retention: {storage.retention_days_cloud} days")

    if not RCLONE_AVAILABLE:
        print("\nERROR: rclone not found. Install with: sudo apt install rclone")
        print("Then configure with: rclone config")
        return

    if not storage.rclone_remote:
        print("\nERROR: No rclone remote configured.")
        print("1. Run 'rclone config' to create a remote")
        print("2. Set cloud_storage.rclone_remote in config/config.json")
        return

    # Verify remote access
    print(f"\nVerifying remote '{storage.rclone_remote}'...")
    success, message = storage.verify_remote()
    if success:
        print(f"  {message}")
    else:
        print(f"  ERROR: {message}")
        return

    # Show status
    status = storage.get_status()
    print(f"\nUpload status:")
    print(f"  Uploaded: {status['uploaded_count']}")
    print(f"  Pending: {status['pending_count']}")
    print(f"  Failed: {status['failed_count']}")

    # Offer to sync
    if status['pending_count'] > 0:
        response = input(f"\nSync {status['pending_count']} pending folders? [y/N] ")
        if response.lower() == 'y':
            print("Syncing...")
            results = await storage.sync_all()
            success_count = sum(1 for v in results.values() if v)
            print(f"Synced {success_count}/{len(results)} folders")

    # Offer to cleanup
    print(f"\nLocal cleanup would delete files older than {storage.retention_days_local} days")
    response = input("Run cleanup? [y/N] ")
    if response.lower() == 'y':
        deleted = storage.cleanup_local()
        print(f"Deleted {deleted} folders")

    print("\nTest complete")


def test_cloud_storage_sync() -> None:
    """Synchronous wrapper for test (CLI entry point)."""
    asyncio.run(test_cloud_storage())


if __name__ == "__main__":
    test_cloud_storage_sync()
