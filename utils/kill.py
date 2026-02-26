#!/usr/bin/env python3
"""
Kill Script for Federated Learning Processes

This script terminates all processes related to the federated learning network
started by run_fl.py, including:
- The main run_fl.py process
- The caffeinate wrapper (if used)
- The Flower server (start_server.py)
- All Flower clients (start_client.py)

Usage:
    python kill.py
    # or make it executable and run directly:
    chmod +x kill.py
    ./kill.py
"""

import os
import subprocess
import signal
import sys


def find_processes(pattern):
    """
    Find all process IDs matching the given pattern.
    
    Args:
        pattern: String pattern to search for in process command lines
        
    Returns:
        List of PIDs (as strings)
    """
    try:
        result = subprocess.run(
            ['pgrep', '-f', pattern],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
        return []
    except Exception as e:
        print(f"Error finding processes for pattern '{pattern}': {e}")
        return []


def kill_processes(pids, signal_type=signal.SIGTERM):
    """
    Kill processes with the given PIDs.
    
    Args:
        pids: List of process IDs (as strings)
        signal_type: Signal to send (default: SIGTERM)
    """
    if not pids or pids == ['']:
        return
    
    for pid in pids:
        try:
            os.kill(int(pid), signal_type)
            print(f"✓ Sent signal to process {pid}")
        except ProcessLookupError:
            print(f"✗ Process {pid} not found (already terminated)")
        except PermissionError:
            print(f"✗ Permission denied for process {pid}")
        except Exception as e:
            print(f"✗ Error killing process {pid}: {e}")


def main():
    """Main function to kill all FL processes"""
    
    print("=" * 70)
    print("KILLING FEDERATED LEARNING PROCESSES")
    print("=" * 70)
    
    # Define patterns for all FL-related processes
    patterns = [
        'federated/synchronous/run_fl.py',
        'federated/synchronous/start_server.py',
        'federated/synchronous/start_client.py',
        'caffeinate.*federated/synchronous',
    ]
    
    all_pids = set()
    
    # Find all matching processes
    print("\nSearching for FL processes...")
    for pattern in patterns:
        pids = find_processes(pattern)
        if pids and pids != ['']:
            print(f"  Found {len(pids)} process(es) matching '{pattern}'")
            all_pids.update(pids)
    
    if not all_pids:
        print("\n✓ No FL processes found. System is clean.")
        return
    
    print(f"\nFound {len(all_pids)} total process(es) to terminate.")
    print("\nTerminating processes (SIGTERM)...")
    kill_processes(list(all_pids), signal.SIGTERM)
    
    # Wait a moment for graceful shutdown
    import time
    time.sleep(2)
    
    # Check if any processes are still alive and force kill them
    print("\nChecking for remaining processes...")
    remaining_pids = set()
    for pattern in patterns:
        pids = find_processes(pattern)
        if pids and pids != ['']:
            remaining_pids.update(pids)
    
    if remaining_pids:
        print(f"\n{len(remaining_pids)} process(es) still running. Force killing (SIGKILL)...")
        kill_processes(list(remaining_pids), signal.SIGKILL)
        time.sleep(1)
    
    # Final check
    final_check = set()
    for pattern in patterns:
        pids = find_processes(pattern)
        if pids and pids != ['']:
            final_check.update(pids)
    
    print("\n" + "=" * 70)
    if final_check:
        print(f"⚠ WARNING: {len(final_check)} process(es) could not be killed:")
        for pid in final_check:
            print(f"  PID: {pid}")
        print("You may need to kill these manually with 'kill -9 <PID>'")
    else:
        print("✓ ALL FL PROCESSES TERMINATED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nKill script interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
