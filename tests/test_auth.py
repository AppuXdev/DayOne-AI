import unittest

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


if __name__ == "__main__":
    unittest.main()
