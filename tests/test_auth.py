import unittest
import os
import subprocess
from pathlib import Path

from fastapi import HTTPException

from main import authenticate_user


class AuthenticationTests(unittest.TestCase):
    def test_employee_credentials_are_valid(self) -> None:
        user = authenticate_user("john_doe", "password123")
        self.assertEqual(user.username, "john_doe")
        self.assertEqual(user.organization, "org_acme")
        self.assertEqual(user.role, "employee")

    def test_admin_credentials_are_valid(self) -> None:
        user = authenticate_user("admin_acme", "password123")
        self.assertEqual(user.username, "admin_acme")
        self.assertEqual(user.organization, "org_acme")
        self.assertEqual(user.role, "admin")

    def test_invalid_password_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            authenticate_user("john_doe", "wrong-password")
        self.assertEqual(ctx.exception.status_code, 401)


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
