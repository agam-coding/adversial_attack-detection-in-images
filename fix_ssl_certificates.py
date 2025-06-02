#!/usr/bin/env python3
import os
import sys

def main():
    """Fix SSL certificate verification on macOS."""
    # Check if we're on macOS
    if sys.platform != 'darwin':
        print("This script is intended for macOS only.")
        return
    
    # Get Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    # Path to the Install Certificates.command script
    cert_script_path = f"/Applications/Python {python_version}/Install Certificates.command"
    alt_cert_script_path = f"/Library/Frameworks/Python.framework/Versions/{python_version}/Resources/Install Certificates.command"
    
    # Check if either path exists
    if os.path.exists(cert_script_path):
        print(f"Running certificate installation script from: {cert_script_path}")
        os.system(f"sh \"{cert_script_path}\"")
    elif os.path.exists(alt_cert_script_path):
        print(f"Running certificate installation script from: {alt_cert_script_path}")
        os.system(f"sh \"{alt_cert_script_path}\"")
    else:
        # If neither exists, provide manual instructions
        print("Could not find the certificate installation script.")
        print("\nManual solution:")
        print("1. Run this command in your terminal:")
        print(f"   /Applications/Python\\ {python_version}/Install\\ Certificates.command")
        print("\nAlternative solution:")
        print("Add this code at the beginning of your script:")
        print("import ssl")
        print("ssl._create_default_https_context = ssl._create_unverified_context")
        
    print("\nAfter fixing the certificates, try running your code again.")

if __name__ == "__main__":
    main()