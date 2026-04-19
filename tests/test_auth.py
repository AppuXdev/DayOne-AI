import unittest
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from main import authenticate_user_db_only


class AuthenticationTests(unittest.TestCase):
    def test_employee_credentials_are_valid(self) -> None:
        with patch("main.auth_db.authenticate_user", return_value={
            "username": "john_doe",
            "organization": "org_acme",
            "role": "employee",
            "tenant_id": "00000000-0000-0000-0000-000000000001",
        }):
            user = authenticate_user_db_only("john_doe", "password123", "org_acme")
        self.assertEqual(user.username, "john_doe")
        self.assertEqual(user.organization, "org_acme")
        self.assertEqual(user.role, "employee")

    def test_admin_credentials_are_valid(self) -> None:
        with patch("main.auth_db.authenticate_user", return_value={
            "username": "admin_acme",
            "organization": "org_acme",
            "role": "admin",
            "tenant_id": "00000000-0000-0000-0000-000000000001",
        }):
            user = authenticate_user_db_only("admin_acme", "password123", "org_acme")
        self.assertEqual(user.username, "admin_acme")
        self.assertEqual(user.organization, "org_acme")
        self.assertEqual(user.role, "admin")

    def test_invalid_password_is_rejected(self) -> None:
        with patch("main.auth_db.authenticate_user", return_value=None):
            with self.assertRaises(HTTPException) as ctx:
                authenticate_user_db_only("john_doe", "wrong-password", "org_acme")
        self.assertEqual(ctx.exception.status_code, 401)

    def test_missing_organization_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            authenticate_user_db_only("john_doe", "password123", "")
        self.assertEqual(ctx.exception.status_code, 400)


class RunScriptSmokeTests(unittest.TestCase):
    def test_run_ps1_is_ascii_and_contains_retry_guard(self) -> None:
        script_path = Path("run.ps1")
        script_text = script_path.read_text(encoding="utf-8")
        self.assertTrue(all(ord(ch) < 128 for ch in script_text))
        self.assertIn("Invoke-IngestWithRetry", script_text)
        self.assertIn("Stop-DayOneJobs", script_text)

    @unittest.skipUnless(os.name == "nt", "PowerShell smoke check is Windows-only")
    def test_run_ps1_parses_in_powershell(self) -> None:
        command = "[scriptblock]::Create((Get-Content -Raw .\\run.ps1)) | Out-Null"
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
