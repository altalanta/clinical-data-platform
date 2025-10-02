#!/usr/bin/env python3
"""
Bootstrap script to create admin users with secure password hashes.
This script should be run once during initial deployment.
"""

import os
import secrets
import getpass
from passlib.context import CryptContext


def generate_secure_password(length: int = 16) -> str:
    """Generate a cryptographically secure password."""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.hash(password)


def main():
    """Bootstrap admin user creation."""
    print("Clinical Data Platform - Admin User Bootstrap")
    print("=" * 50)
    
    # Check if already configured
    if os.getenv("ADMIN_PASSWORD_HASH"):
        print("Admin user already configured via ADMIN_PASSWORD_HASH")
        print("If you need to reset, unset the environment variable and re-run this script")
        return
    
    # Get admin details
    admin_email = input("Admin email [admin@clinical-platform.com]: ").strip()
    if not admin_email:
        admin_email = "admin@clinical-platform.com"
    
    print("\nPassword options:")
    print("1. Generate secure random password")
    print("2. Enter custom password")
    
    choice = input("Choice (1/2) [1]: ").strip()
    if choice == "2":
        password = getpass.getpass("Enter admin password: ")
        confirm_password = getpass.getpass("Confirm password: ")
        
        if password != confirm_password:
            print("ERROR: Passwords do not match")
            return
        
        if len(password) < 12:
            print("ERROR: Password must be at least 12 characters")
            return
    else:
        password = generate_secure_password()
        print(f"Generated password: {password}")
        print("SAVE THIS PASSWORD - it will not be shown again!")
    
    # Hash password
    password_hash = hash_password(password)
    
    # Generate JWT secret if needed
    jwt_secret = os.getenv("JWT_SECRET") or secrets.token_urlsafe(32)
    
    print("\n" + "=" * 50)
    print("Bootstrap complete! Set these environment variables:")
    print("=" * 50)
    print(f"export ADMIN_EMAIL='{admin_email}'")
    print(f"export ADMIN_PASSWORD_HASH='{password_hash}'")
    if not os.getenv("JWT_SECRET"):
        print(f"export JWT_SECRET='{jwt_secret}'")
    print("export ENVIRONMENT='prod'")
    print("=" * 50)
    
    # Optionally write to .env file
    create_env = input("\nCreate .env.bootstrap file? (y/N): ").lower() == 'y'
    if create_env:
        with open(".env.bootstrap", "w") as f:
            f.write(f"ADMIN_EMAIL={admin_email}\n")
            f.write(f"ADMIN_PASSWORD_HASH={password_hash}\n")
            if not os.getenv("JWT_SECRET"):
                f.write(f"JWT_SECRET={jwt_secret}\n")
            f.write("ENVIRONMENT=prod\n")
        print("Environment variables written to .env.bootstrap")
        print("SECURITY WARNING: This file contains sensitive credentials!")
        print("Move it to a secure location and set proper file permissions (600)")


if __name__ == "__main__":
    main()