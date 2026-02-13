#!/usr/bin/env python3
"""
Setup NASA Earthdata Authentication
====================================

This script helps you configure authentication for downloading OCO-2 data
from NASA's GES DISC and MODIS data from LAADS DAAC.

It will:
1. Prompt for your NASA Earthdata credentials
2. Create/update your ~/.netrc file for automatic authentication
3. Set up environment variables for LAADS token

Usage:
    python workspace/setup_auth.py
"""

import os
import sys
from pathlib import Path
import getpass
import stat


def setup_netrc():
    """Create or update .netrc file for NASA Earthdata authentication."""
    
    print("=" * 70)
    print("NASA Earthdata Login Setup")
    print("=" * 70)
    print()
    print("To download OCO-2 data, you need a NASA Earthdata account.")
    print("Register at: https://urs.earthdata.nasa.gov/users/new")
    print()
    
    # Get credentials
    username = input("Enter your Earthdata username: ").strip()
    if not username:
        print("❌ Username required")
        return False
    
    password = getpass.getpass("Enter your Earthdata password: ")
    if not password:
        print("❌ Password required")
        return False
    
    # Create .netrc file
    netrc_path = Path.home() / '.netrc'
    
    # Read existing .netrc if it exists
    existing_lines = []
    if netrc_path.exists():
        with open(netrc_path, 'r') as f:
            existing_lines = f.readlines()
    
    # Remove old NASA entries
    filtered_lines = []
    skip_until_next = False
    for line in existing_lines:
        if 'urs.earthdata.nasa.gov' in line or 'oco2.gesdisc.eosdis.nasa.gov' in line:
            skip_until_next = True
            continue
        if skip_until_next:
            if line.strip() and not line.strip().startswith(('login', 'password')):
                skip_until_next = False
            else:
                continue
        filtered_lines.append(line)
    
    # Add NASA Earthdata entries
    netrc_content = ''.join(filtered_lines)
    
    # Add entries for different NASA servers
    netrc_content += f"\nmachine urs.earthdata.nasa.gov\n"
    netrc_content += f"    login {username}\n"
    netrc_content += f"    password {password}\n"
    netrc_content += f"\nmachine oco2.gesdisc.eosdis.nasa.gov\n"
    netrc_content += f"    login {username}\n"
    netrc_content += f"    password {password}\n"
    
    # Write .netrc
    with open(netrc_path, 'w') as f:
        f.write(netrc_content)
    
    # Set permissions (must be 600 for .netrc)
    os.chmod(netrc_path, stat.S_IRUSR | stat.S_IWUSR)
    
    print(f"\n✓ Created/updated {netrc_path}")
    print(f"  Permissions set to: 600 (owner read/write only)")
    
    return True


def setup_env_vars():
    """Show how to set up environment variables."""
    
    print("\n" + "=" * 70)
    print("Environment Variables Setup")
    print("=" * 70)
    print()
    print("For MODIS data downloads, you need a LAADS DAAC token.")
    print("Get your token at: https://ladsweb.modaps.eosdis.nasa.gov/profile/")
    print()
    
    laads_token = input("Enter your LAADS token (or press Enter to skip): ").strip()
    
    shell = os.environ.get('SHELL', '/bin/bash')
    if 'zsh' in shell:
        rc_file = '~/.zshrc'
    else:
        rc_file = '~/.bashrc'
    
    print(f"\n📝 Add these lines to your {rc_file}:")
    print("-" * 70)
    
    if laads_token:
        print(f"export LAADS_TOKEN='{laads_token}'")
    else:
        print("export LAADS_TOKEN='your_token_here'")
    
    print("\n# Optional: Set custom data storage paths")
    print("# export OCO2_DATA_ROOT='/path/to/your/data'")
    print("# export CURC_DATA_ROOT='/projects/${USER}/oco2_data'")
    print("-" * 70)
    
    return True


def test_authentication():
    """Test the authentication setup."""
    
    print("\n" + "=" * 70)
    print("Testing Authentication")
    print("=" * 70)
    print()
    
    import requests
    from requests.auth import HTTPBasicAuth
    
    # Load credentials from .netrc
    netrc_path = Path.home() / '.netrc'
    if not netrc_path.exists():
        print("❌ .netrc file not found")
        return False
    
    # Parse .netrc
    username = None
    password = None
    with open(netrc_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'urs.earthdata.nasa.gov' in line:
                # Find login and password in next few lines
                for j in range(i+1, min(i+5, len(lines))):
                    if 'login' in lines[j]:
                        username = lines[j].split()[1]
                    elif 'password' in lines[j]:
                        password = lines[j].split()[1]
                break
    
    if not username or not password:
        print("❌ Could not parse .netrc file")
        return False
    
    print(f"Testing with username: {username}")
    
    try:
        # Test Earthdata authentication
        session = requests.Session()
        session.auth = (username, password)
        
        response = session.get(
            "https://urs.earthdata.nasa.gov/api/users/user",
            timeout=10
        )
        
        if response.status_code == 200:
            print("✓ Earthdata authentication successful!")
            user_info = response.json()
            print(f"  Logged in as: {user_info.get('first_name', '')} {user_info.get('last_name', '')}")
            print(f"  Email: {user_info.get('email_address', '')}")
            return True
        else:
            print(f"❌ Authentication failed with status: {response.status_code}")
            print("   Please check your credentials and try again")
            return False
            
    except Exception as e:
        print(f"❌ Error testing authentication: {e}")
        return False


def main():
    """Main setup function."""
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "NASA Earthdata Authentication Setup" + " " * 18 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Setup .netrc
    if not setup_netrc():
        print("\n❌ Setup failed")
        return 1
    
    # Setup environment variables
    setup_env_vars()
    
    # Test authentication
    if test_authentication():
        print("\n" + "=" * 70)
        print("✓ Setup Complete!")
        print("=" * 70)
        print()
        print("You can now run:")
        print("  python workspace/demo_phase_02.py --dry-run --date 2018-10-18")
        print()
        print("Or in Python:")
        print("  from src.phase_02_ingestion import DataIngestionManager")
        print("  manager = DataIngestionManager()")
        print()
        return 0
    else:
        print("\n⚠️  Setup completed, but authentication test failed")
        print("   Check your credentials and try running setup again")
        return 1


if __name__ == "__main__":
    sys.exit(main())
