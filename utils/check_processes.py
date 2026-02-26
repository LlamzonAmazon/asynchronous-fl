#!/usr/bin/env python3
"""
Check Script for Federated Learning Processes

This script checks for any processes related to the federated learning network
started by run_fl.py, including:
- The main run_fl.py process
- The caffeinate wrapper (if used)
- The Flower server (start_server.py)
- All Flower clients (start_client.py)

Usage:
    python utils/check_processes.py
"""

import subprocess
import sys


def find_processes(pattern):
    """
    Find all process IDs matching the given pattern.
    
    Args:
        pattern: String pattern to search for in process command lines
        
    Returns:
        List of tuples (PID, command) or empty list
    """
    processes = []
    
    # Try pgrep first (more reliable on macOS)
    try:
        result = subprocess.run(
            ['pgrep', '-fl', pattern],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        processes.append((parts[0], parts[1]))
            return processes
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        pass
    except Exception:
        pass
    
    # Fallback to ps if pgrep didn't work
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Skip header
                for line in lines[1:]:
                    if pattern in line:
                        parts = line.split(None, 10)
                        if len(parts) >= 11:
                            pid = parts[1]
                            cmd = ' '.join(parts[10:])
                            processes.append((pid, cmd))
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        pass
    except Exception:
        pass
    
    return processes


def main():
    """Main function to check for FL processes"""
    
    print("=" * 70)
    print("CHECKING FOR FEDERATED LEARNING PROCESSES")
    print("=" * 70)
    
    # Define patterns for all FL-related processes
    patterns = [
        ('federated/synchronous/run_fl.py', 'Main FL script'),
        ('federated/synchronous/start_server.py', 'Flower server'),
        ('federated/synchronous/start_client.py', 'Flower clients'),
        ('caffeinate.*federated/synchronous', 'Caffeinate wrapper'),
    ]
    
    all_processes = []
    
    # Find all matching processes
    print("\nSearching for FL processes...\n")
    for pattern, description in patterns:
        processes = find_processes(pattern)
        if processes:
            print(f"  {description} ({len(processes)} process(es)):")
            for pid, cmd in processes:
                print(f"    PID {pid}: {cmd[:80]}")
            all_processes.extend(processes)
        else:
            print(f"  {description}: None found")
    
    print("\n" + "=" * 70)
    if all_processes:
        print(f"⚠ WARNING: Found {len(all_processes)} FL-related process(es) still running!")
        print("\nTo kill these processes, run:")
        print("  python utils/kill.py")
        print("=" * 70)
        sys.exit(1)
    else:
        print("✓ No FL processes found. System is clean.")
        print("=" * 70)
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCheck script interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
