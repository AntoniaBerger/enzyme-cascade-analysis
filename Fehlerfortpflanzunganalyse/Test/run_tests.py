"""
Einfacher Test-Runner für data_hadler Tests

Führt alle Tests aus und zeigt Ergebnisse an.
"""

import sys
import os
import subprocess

def run_tests():
    """Führe die Tests aus"""
    print("="*60)
    print(" DATA_HANDLER TESTS")
    print("="*60)
    
    try:
        # Führe die Tests aus
        result = subprocess.run([
            sys.executable, 
            'test_data_hadler.py'
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ Alle Tests erfolgreich!")
        else:
            print("❌ Einige Tests fehlgeschlagen!")
            
    except Exception as e:
        print(f"Fehler beim Ausführen der Tests: {e}")

if __name__ == '__main__':
    run_tests()
