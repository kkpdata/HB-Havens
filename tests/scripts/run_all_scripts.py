#!/usr/bin/env python3
"""
Run all script_*_simple.py scripts to test the optimized simple diffraction calculations.
"""

import subprocess
import sys
import os
import time
import glob
import logging

# Add trunk directory to path for logger access
trunkdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
tempdir = os.path.join(os.path.dirname(trunkdir), 'temp')

if trunkdir not in sys.path:
    sys.path.append(trunkdir)

if not os.path.exists(tempdir):
    os.makedirs(tempdir)

# Initialize logging
from hbhavens.core.logger import initialize_logger
initialize_logger(filename=os.path.join(tempdir, 'hbhavens.log'))
logger = logging.getLogger()

def run_script(script_path):
    """Run a single script and measure execution time."""
    script_name = os.path.basename(script_path)
    logger.info(f'RUNNING Script: {script_name}')
    
    try:
        start_time = time.time()
        
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, 
                              cwd=os.path.dirname(script_path))
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f'{script_name} execution time: {execution_time:.1f}s')
        
        if result.returncode == 0:
            logger.info(f'SUCCESS: {script_name}')
        else:
            logger.error(f'FAILED: {script_name} - Return code: {result.returncode}')
            
        if result.stderr:
            logger.warning(f'{script_name} stderr: {result.stderr.strip()[:200]}...')
            
        return result.returncode == 0, execution_time
        
    except Exception as e:
        logger.error(f'ERROR running {script_name}: {e}')
        return False, 0.0

def main():
    """Run lal script_*_simple.py scripts."""
    
    # Find all script_*_simple.py scripts (excluding issue-specific ones)
    scripts_dir = os.path.dirname(__file__)
    script_pattern = os.path.join(scripts_dir, "script_*_simple.py")
    all_scripts = glob.glob(script_pattern)
    
    # Filter out issue-specific scripts
    scripts = [s for s in all_scripts if " issue " not in s]
    scripts.sort()
    
    logger.info(f'START Script Runner - Found {len(scripts)} scripts')
    
    results = {}
    start_total = time.time()
    
    for script_path in scripts:
        script_name = os.path.basename(script_path)
        success, exec_time = run_script(script_path)
        results[script_name] = {'success': success, 'time': exec_time}
    
    end_total = time.time()
    total_time = end_total - start_total
    
    # Summary
    successful = [(name, data) for name, data in results.items() if data['success']]
    failed = [(name, data) for name, data in results.items() if not data['success']]
    
    success_rate = len(successful) / len(results) * 100 if results else 0
    avg_time = sum(data['time'] for data in results.values()) / len(results) if results else 0
    
    logger.info('SCRIPT EXECUTION SUMMARY')
    logger.info(f'Total time: {total_time:.1f}s, Success rate: {success_rate:.1f}%, Avg: {avg_time:.1f}s')
    
    logger.info(f'Successful ({len(successful)}):')
    for name, data in successful:
        logger.info(f'  {name}: {data["time"]:.1f}s')
    
    if failed:
        logger.info(f'Failed ({len(failed)}):')
        for name, data in failed:
            logger.error(f'  {name}: {data["time"]:.1f}s')
    
    # Save detailed results to temp folder
    scripts_dir = os.path.dirname(__file__)
    trunk_dir = os.path.dirname(os.path.dirname(scripts_dir))
    temp_dir = os.path.join(os.path.dirname(trunk_dir), 'temp')
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    logger.info('END Script Runner')

if __name__ == "__main__":
    main()